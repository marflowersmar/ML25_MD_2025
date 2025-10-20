# inference_xgboost.py — aplica el thr_final de VALIDACIÓN, 1 submission 0/1

from pathlib import Path
import sys, joblib, numpy as np, pandas as pd
from datetime import datetime

CURRENT_FILE = Path(__file__).resolve()
sys.path.append(str(CURRENT_FILE.parent.parent))

from data_processing import read_test_data, read_csv
from model_xgboost import XGBoostModel  # compat al cargar joblib

MODELS_DIR  = CURRENT_FILE.parent.parent / "trained_models"
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _latest_model():
    cands = sorted(MODELS_DIR.glob("xgb_*.pkl"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No hay modelos XGBoost en {MODELS_DIR}")
    return cands[-1]

def _read_thr(model_path: Path, suffix: str):
    f = model_path.with_name(model_path.stem + suffix)
    if f.exists():
        try: return float(f.read_text().strip())
        except: return None
    return None

if __name__ == "__main__":
    print("="*80); print("XGBoost Inference (umbral de validación)"); print("="*80)

    model_path = _latest_model()
    print(f"Modelo: {model_path.name}")
    model = joblib.load(model_path)

    # IDs en orden original
    test_raw = read_csv("customer_purchases_test")
    ids = test_raw["purchase_id"].reset_index(drop=True)

    X_test = read_test_data()
    proba = model.predict_proba(X_test)[:, 1].astype(float)

    thr = (_read_thr(model_path, "_thr_final.txt")
           or _read_thr(model_path, "_thr_youden.txt")
           or _read_thr(model_path, "_thr_f1.txt")
           or 0.5)
    thr = float(np.clip(thr, 0.01, 0.99))
    print(f"Umbral aplicado (de VALIDACIÓN): {thr:.4f}")

    pred = (proba >= thr).astype(int)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = f"submission_{model_path.stem}_thr{thr:.4f}_{ts}.csv"
    outpath = RESULTS_DIR / outname
    pd.DataFrame({"purchase_id": ids, "prediction": pred}).to_csv(outpath, index=False)

    n = len(pred); pos = int(pred.sum()); neg = n - pos
    print(f"✓ Submission guardada en: {outpath}")
    print(f"Total: {n}")
    print(f"Positivos: {pos} ({pos/n*100:.2f}%)  |  Negativos: {neg} ({neg/n*100:.2f}%)")
    print("="*80)
