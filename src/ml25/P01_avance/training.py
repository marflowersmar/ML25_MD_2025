import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

# ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
import json

# Custom
from data_processing import read_train_data  # tu propio script
from model import PurchaseModel

# Logs opcionales
def setup_logger(name="training_log"):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ======================
# Funciones auxiliares
# ======================
def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    return data


def _evaluate_at_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    roc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1_vals = 2 * (prec * rec) / np.clip(prec + rec, 1e-12, None)
    idx = int(np.nanargmax(f1_vals))
    best_thr = float(thr[idx - 1]) if idx > 0 else 0.5
    best_f1 = float(f1_vals[idx])
    y_pred = (y_proba >= best_thr).astype(int)
    return {
        "roc_auc": float(roc),
        "pr_auc": float(ap),
        "best_f1": best_f1,
        "best_threshold": best_thr,
        "f1_at_best_thr": float(f1_score(y_true, y_pred)),
    }


# ======================
# Entrenamiento
# ======================
def run_training(X, y, classifier: str):
    logger = setup_logger(f"training_{classifier}")
    logger.info("Inicio de entrenamiento...")

    # 1. Separar en entrenamiento y validación
    X = _to_numeric_df(X)
    y = pd.Series(y).astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split -> X_train: {X_tr.shape}, X_val: {X_va.shape}")

    # 2. Entrenar el modelo
    model = PurchaseModel()
    model.fit(X_tr, y_tr)
    logger.info("Modelo entrenado correctamente")

    # 3. Validación
    proba_va = model.predict_proba(X_va)
    metrics = _evaluate_at_best_threshold(y_va.values, proba_va)
    best_thr = metrics["best_threshold"]
    y_pred = (proba_va >= best_thr).astype(int)
    report = classification_report(y_va, y_pred, digits=4)
    cm = confusion_matrix(y_va, y_pred)

    # 4. Guardar métricas y modelo
    ARTIFACTS_DIR = Path(__file__).resolve().parent
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(ARTIFACTS_DIR / "metrics_final.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame({"y_true": y_va, "y_proba": proba_va}).to_csv(
        ARTIFACTS_DIR / "val_predictions.csv", index=False
    )
    with open(ARTIFACTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    model_path = model.save(prefix=classifier)
    logger.info(f"Métricas: {metrics}")
    logger.info(f"Modelo guardado en: {model_path}")
    logger.info("Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    X, y = read_train_data()
    run_training(X, y, classifier="final_gradient_boost")
