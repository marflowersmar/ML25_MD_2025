# inference_rf.py
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------------------------------
# Rutas y path hacking compatible con tu layout de proyecto
# ------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJ_DIR = CURRENT_FILE.parent.parent   # src/ml25/P01_avance
sys.path.append(str(PROJ_DIR))

from data_processing import read_csv, read_test_data
from model_rf import RandomForestModel  # opcional si el objeto guardado es el wrapper

MODELS_DIR = PROJ_DIR / "trained_models"
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
def _list_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("rf_*.pkl"), key=lambda p: p.stat().st_mtime)

def _load_any(pkl_path: Path):
    """Carga el objeto guardado: wrapper RandomForestModel o sklearn RF."""
    return joblib.load(pkl_path)

def _read_threshold_sidecars(model_path: Path):
    """
    Lee *_thr_f1.txt, *_thr_balanced.txt, *_thr_safe.txt si existen.
    """
    base = Path(str(model_path).replace(".pkl", ""))
    files = {
        "f1":       base.with_name(base.name + "_thr_f1.txt"),
        "balanced": base.with_name(base.name + "_thr_balanced.txt"),
        "safe":     base.with_name(base.name + "_thr_safe.txt"),
    }
    out = {}
    for k, f in files.items():
        if f.exists():
            try:
                out[k] = float(f.read_text().strip())
            except Exception:
                pass
    return out

def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------------------------------------------------------------
# Inference principal
# ------------------------------------------------------------------
def run_inference(target_positive_rate: float | None = None):
    print("=" * 80)
    print("INFERENCIA RF - PIPELINE CONSISTENTE CON ENTRENAMIENTO")
    print("=" * 80)

    # 1) Modelo más reciente
    models = _list_models()
    if not models:
        raise FileNotFoundError(f"No hay modelos en: {MODELS_DIR}")
    model_path = models[-1]
    print(f"Modelo encontrado: {model_path.name}")
    model_obj = _load_any(model_path)

    # 2) Cargar test preprocesado igual que en entrenamiento
    print("Cargando y preprocesando test con read_test_data()...")
    X_test = read_test_data()                 # DF numérico final
    test_raw = read_csv("customer_purchases_test")
    purchase_ids = test_raw["purchase_id"].values
    print(f"X_test.shape = {X_test.shape}")

    # 3) Probabilidades
    if hasattr(model_obj, "predict_proba"):
        y_proba = model_obj.predict_proba(X_test)[:, 1]
    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "predict_proba"):
        y_proba = model_obj.model.predict_proba(X_test)[:, 1]
    else:
        y_pred_hard = model_obj.predict(X_test)
        y_proba = np.clip(y_pred_hard.astype(float), 0.0, 1.0)

    # 4) Selección de threshold: F1 -> BALANCED -> SAFE -> tasa objetivo -> 0.5
    side = _read_threshold_sidecars(model_path)
    thr = None
    origin = ""

    candidates = []
    if "f1" in side:        candidates.append(("thr_f1", side["f1"]))
    if "balanced" in side:  candidates.append(("thr_balanced", side["balanced"]))
    if "safe" in side:      candidates.append(("thr_safe", side["safe"]))

    if candidates:
        origin, thr = candidates[0]
    elif target_positive_rate is not None:
        q = 1.0 - float(target_positive_rate)
        q = min(max(q, 0.0), 1.0)
        thr = float(np.quantile(y_proba, q))
        origin = f"quantile@{target_positive_rate:.3f}"
    else:
        thr = 0.5
        origin = "default_0.5"

    # Clip preventivo
    thr = float(min(max(thr, 0.01), 0.99))

    # 5) Predicción binaria y submission
    y_pred = (y_proba >= thr).astype(int)
    pos_rate = y_pred.mean()
    print(f"Threshold usado: {thr:.4f}  [{origin}]  | PosRate: {pos_rate:.3f}")

    sub = pd.DataFrame({"purchase_id": purchase_ids, "prediction": y_pred.astype(int)},
                       columns=["purchase_id", "prediction"])
    out_path = RESULTS_DIR / f"submission_{origin}_{thr:.4f}_{_timestamp()}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Submission guardada en: {out_path}")

    # 6) Resumen de probabilidades
    print("-" * 80)
    print("Resumen probabilidades:")
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    qv = np.quantile(y_proba, qs)
    print(f"min={qv[0]:.4f}  p25={qv[1]:.4f}  p50={qv[2]:.4f}  p75={qv[3]:.4f}  max={qv[4]:.4f}")
    print("=" * 80)
    return out_path

if __name__ == "__main__":
    # Para aproximar 419/978 ≈ 0.428 positivos, descomenta:
    # out = run_inference(target_positive_rate=0.428)
    out = run_inference()
