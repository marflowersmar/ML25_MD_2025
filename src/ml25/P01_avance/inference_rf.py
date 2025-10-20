# inference_rf.py
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt  # por si luego quieres graficar
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay  # opcional

from data_processing import read_test_data
from model_rf import PurchaseModel

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
MODELS_DIR = BASE_DIR / "trained_models"
RESULTS_DIR = BASE_DIR / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def find_latest_model(models_dir: Path) -> Path:
    files = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No hay modelos en {models_dir}")
    return files[0]


def load_wrapper(path: Path) -> PurchaseModel:
    obj = joblib.load(path)
    if isinstance(obj, PurchaseModel):
        return obj
    if isinstance(obj, dict) and "wrapper" in obj:
        return obj["wrapper"]
    return PurchaseModel(clf=obj)


def read_threshold(th_file: Path) -> float:
    if th_file.exists():
        try:
            return float(th_file.read_text().strip())
        except Exception:
            pass
    return 0.5


def auto_adjust_threshold(probs: np.ndarray, th_initial: float) -> float:
    """
    Si el threshold inicial produce saturación, se autoajusta con percentiles.
    Regresa un threshold que evita 99 por ciento en una sola clase cuando sea posible.
    """
    th = float(th_initial)
    pos_rate = float((probs >= th).mean())
    # Si está razonable, lo dejamos
    if 0.05 <= pos_rate <= 0.95:
        return th

    # Si casi todo 1, subimos corte a p90 y probamos, si sigue alto, subimos a p95
    if pos_rate > 0.95:
        th_try = float(np.quantile(probs, 0.90))
        if (probs >= th_try).mean() < 0.95:
            return th_try
        th_try = float(np.quantile(probs, 0.95))
        return th_try

    # Si casi todo 0, bajamos a p10 o p05
    if pos_rate < 0.05:
        th_try = float(np.quantile(probs, 0.10))
        if (probs >= th_try).mean() > 0.05:
            return th_try
        th_try = float(np.quantile(probs, 0.05))
        return th_try

    return th


def main():
    # 1) Modelo
    model_path = find_latest_model(MODELS_DIR)
    print(f"📦 Modelo encontrado: {model_path.name}")
    wrapper = load_wrapper(model_path)
    print(f"✅ Cargado: {repr(wrapper)}")

    # 2) Datos
    print("📊 Cargando datos de test...")
    X = read_test_data()
    print(f"📈 Dimensiones test: {X.shape}")

    # 3) Probabilidades
    probs = wrapper.predict_proba(X)
    q = np.percentile(probs, [0, 25, 50, 75, 100])
    print(f"Probas  min={q[0]:.4f}  p25={q[1]:.4f}  mediana={q[2]:.4f}  p75={q[3]:.4f}  max={q[4]:.4f}")

    # 4) Threshold
    th_file = MODELS_DIR / "optimal_threshold.txt"
    th0 = read_threshold(th_file)
    th = auto_adjust_threshold(probs, th0)
    if th != th0:
        print(f"⚠️ Threshold autoajustado de {th0:.4f} a {th:.4f} para evitar saturación")
    else:
        print(f"🔧 Threshold usado: {th:.4f}")

    # 5) Predicción
    preds = (probs >= th).astype(int)
    uniq, cnts = np.unique(preds, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnts)}
    print(f"📊 Distribución predicciones: {dist}")

    # 6) Guardar resultados
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = MODELS_DIR / f"submission_{ts}.csv"
    kaggle_df = pd.DataFrame({
        "ID": np.arange(len(X), dtype=int),
        "prediction": preds.astype(int),
    })
    kaggle_df.to_csv(out_csv, index=False)
    print(f"✅ Saved predictions to {out_csv}")

    # Guardar archivo con probabilidades para revisión
    out_probs = RESULTS_DIR / f"submission_probs_{ts}.csv"
    full_df = pd.DataFrame({
        "ID": np.arange(len(X), dtype=int),
        "prediction": preds.astype(int),
        "probability": probs.astype(float),
    })
    full_df.to_csv(out_probs, index=False)
    print(f"✅ Saved predictions with probs to {out_probs}")

    print("🎉 PROCESO COMPLETADO")


if __name__ == "__main__":
    main()
