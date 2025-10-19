# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def get_config(self):
        return {
            "model": "XGBClassifier",
            **self.clf.get_params(),
        }

    def __repr__(self):
        cfg = self.get_config()
        core = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "scale_pos_weight": cfg["scale_pos_weight"],
            "random_state": cfg["random_state"],
        }
        return f"<PurchaseModel XGB {core}>"

    def save(self, prefix: str):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)
        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
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
