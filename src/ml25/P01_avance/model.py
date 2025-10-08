# model.py
# Modelo óptimo para el Proyecto 1 (entrenamiento para Kaggle)
# Mariana Flores Martínez - ICE V - CETYS Universidad

from pathlib import Path
from datetime import datetime
import joblib
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier

CURRENT_FILE = Path(__file__).resolve()
LOCAL_DIR = CURRENT_FILE.parent
MODELS_DIR = LOCAL_DIR / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    return data


def _load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    path = LOCAL_DIR / "train_df_full.csv"
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("El dataset no contiene columna 'label'")
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    return _to_numeric_df(X), y


def _evaluate_threshold(y_true, y_proba):
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


class PurchaseModel:
    def __init__(self):
        self.name = "GradientBoostingClassifier"
        self.model = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_config(self):
        return self.model.get_params()

    def save(self, prefix="gradient_boost"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        path = MODELS_DIR / filename
        joblib.dump(self.model, path)
        print(f"Modelo guardado en: {path}")
        return path


def train_final_model():
    X, y = _load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = PurchaseModel()
    model.fit(X_train, y_train)
    proba_val = model.predict_proba(X_val)

    metrics = _evaluate_threshold(y_val.values, proba_val)
    model_path = model.save("final_gradient_boost")

    metrics_path = LOCAL_DIR / "metrics_final.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nResultados del modelo final:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nModelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    train_final_model()
