# inference_rf.py — Usa SIEMPRE el modelo fijo por nombre, portable entre equipos
from pathlib import Path
import sys, os, argparse, joblib, hashlib
import pandas as pd
import numpy as np
from datetime import datetime


CURRENT_FILE = Path(__file__).resolve()
PROJ_DIR = CURRENT_FILE.parent.parent
sys.path.append(str(PROJ_DIR))

from data_processing import read_csv, read_test_data
from model_rf import RandomForestModel  # compat si el objeto guardado es wrapper

MODELS_DIR = PROJ_DIR / "trained_models"
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Nombre exacto del modelo "oficial"
PINNED_MODEL_FILENAME = "rf_n100_md6_mss50_msl25_mf0.2_cwbal_ULTRA_SAFE_20251020_105334.pkl"

def _timestamp():
    fixed_date = datetime(2025, 9, 21, 0, 0, 0)
    return fixed_date.strftime("%Y%m%d_%H%M%S")

def _list_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("rf_*.pkl"), key=lambda p: p.stat().st_mtime)

def _find_pinned_model() -> Path | None:
    """
    Busca el archivo con nombre exacto PINNED_MODEL_FILENAME en trained_models/** recursivo.
    Si hay varios, devuelve el más reciente por mtime.
    """
    if not MODELS_DIR.exists():
        return None
    candidates = list(MODELS_DIR.rglob(PINNED_MODEL_FILENAME))
    if not candidates:
        return None
    # si hay varias copias en distintas máquinas/rutas relativas, tomar la más reciente
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

def _load_any(pkl_path: Path):
    return joblib.load(pkl_path)

def _read_threshold_sidecars(model_path: Path):
    """
    Prioriza thresholds del mismo folder del modelo fijo.
    Mantiene compat con tu esquema de sidecars: _thr_f1/_thr_balanced/_thr_safe.
    """
    base = Path(str(model_path).replace(".pkl", ""))
    files = {
        "f1":       base.with_name(base.name + "_thr_f1.txt"),
        "balanced": base.with_name(base.name + "_thr_balanced.txt"),
        "safe":     base.with_name(base.name + "_thr_safe.txt"),
        "lower":    base.with_name(base.name + "_thr_lower.txt"),
    }
    out = {}
    for k, f in files.items():
        if f.exists():
            try:
                out[k] = float(f.read_text().strip())
            except Exception:
                pass
    return out

def run_inference(target_positive_rate: float | None = None):
    print("=" * 80)
    print("INFERENCIA RF - MODELO FIJO POR NOMBRE (PORTABLE)")
    print("=" * 80)

    # 1) Buscar el modelo por nombre exacto, portable entre equipos
    model_path = _find_pinned_model()
    if model_path is None:
        print(f"[AVISO] No encontré {PINNED_MODEL_FILENAME} en {MODELS_DIR}/**")
        # contingencia: usar el último rf_*.pkl si existe
        models = _list_models()
        if not models:
            raise FileNotFoundError(f"No hay modelos en: {MODELS_DIR}")
        model_path = models[-1]
        print(f"[CONTINGENCIA] Usando el más reciente: {model_path.name}")

    print(f"Modelo usado: {model_path.relative_to(PROJ_DIR)}")
    model_obj = _load_any(model_path)

    # 2) Cargar test con el mismo pipeline que training
    print("Cargando y preprocesando test con read_test_data()...")
    X_test = read_test_data()  # DF numérico final
    test_raw = read_csv("customer_purchases_test")
    ids = test_raw["purchase_id"].values
    print(f"X_test.shape = {X_test.shape}")

    # 3) Probabilidades
    if hasattr(model_obj, "predict_proba"):
        y_proba = model_obj.predict_proba(X_test)[:, 1]
    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "predict_proba"):
        y_proba = model_obj.model.predict_proba(X_test)[:, 1]
    else:
        y_pred_hard = model_obj.predict(X_test)
        y_proba = np.clip(y_pred_hard.astype(float), 0.0, 1.0)


    # 4) Thresholds: sidecars del modelo fijo primero; si faltan, usar tasa objetivo; si no, 0.5
    side = _read_threshold_sidecars(model_path)
    thr = None; origin = ""
    for key in ["f1", "lower", "balanced", "safe"]:
        if key in side:
            thr = side[key]; origin = f"thr_{key}"; break
    if thr is None and target_positive_rate is not None:
        q = 1.0 - float(target_positive_rate)
        thr = float(np.quantile(y_proba, np.clip(q, 0, 1))); origin = f"quantile@{target_positive_rate:.3f}"
    if thr is None:
        thr = 0.5; origin = "default_0.5"
    thr = float(np.clip(thr, 0.01, 0.99))

    # 5) Predicción binaria y submission
    y_pred = (y_proba >= thr).astype(int)
    pos_rate = y_pred.mean()
    print(f"Threshold usado: {thr:.4f}  [{origin}]  | PosRate: {pos_rate:.3f}")

    sub = pd.DataFrame({"ID": ids, "prediction": y_pred.astype(int)}, columns=["ID", "prediction"])
    out_path = RESULTS_DIR / f"submission_{origin}_{thr:.4f}_{_timestamp()}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Submission guardada en: {out_path}")

    # 6) Resumen probabilidades
    print("-" * 80)
    print("Resumen probabilidades:")
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    qv = np.quantile(y_proba, qs)
    print(f"min={qv[0]:.4f}  p25={qv[1]:.4f}  p50={qv[2]:.4f}  p75={qv[3]:.4f}  max={qv[4]:.4f}")
    print("=" * 80)
    return out_path

if __name__ == "__main__":
    # Sin argumentos: siempre intenta el modelo fijo por nombre
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_positive_rate", type=float, default=None, help="Opcional: fijar tasa de 1s")
    args = ap.parse_args()
    run_inference(target_positive_rate=args.target_positive_rate)
