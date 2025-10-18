# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float = "sqrt",
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # ---- API principal ----
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        # Columna 1 es la probabilidad de la clase positiva (label=1)
        return self.clf.predict_proba(X)[:, 1]

    def get_config(self):
        return {
            "model": "RandomForestClassifier",
            **self.clf.get_params(),
        }

    def __repr__(self):
        cfg = self.get_config()
        core = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "max_features": cfg["max_features"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "class_weight": cfg["class_weight"],
            "random_state": cfg["random_state"],
        }
        return f"<PurchaseModel RF {core}>"

    # ---- Persistencia ----
    def save(self, prefix: str):
        """
        Guarda el modelo en MODELS_DIR con nombre:
        <prefix>_<timestamp>.pkl
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)
        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        """
        Carga el modelo desde MODELS_DIR/filename
        """
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"{self.__repr__} || Model loaded from {filepath}")
        return model


# ---- Utilidades de entrenamiento/evaluación (opcionales) ----
def evaluate_model(model: "PurchaseModel", X, y, title: str = "RF"):
    prob = model.predict_proba(X)
    pred = (prob >= 0.5).astype(int)
    roc = roc_auc_score(y, prob)
    pr = average_precision_score(y, prob)
    print(f"[{title}] ROC-AUC: {roc:.5f} | PR-AUC: {pr:.5f}")
    print(classification_report(y, pred, digits=4))
    return {"roc_auc": roc, "pr_auc": pr}
