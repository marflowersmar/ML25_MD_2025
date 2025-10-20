# model_rf.py
from pathlib import Path
from datetime import datetime
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.model_name = self.__class__.__name__

    def fit(self, X, y):
        X_clean = X.fillna(0)
        if hasattr(X_clean, "values"):
            X_clean = X_clean.values
        self.model.fit(X_clean, y)
        return self

    def predict(self, X):
        X_clean = X.fillna(0)
        if hasattr(X_clean, "values"):
            X_clean = X_clean.values
        return self.model.predict(X_clean)

    def predict_proba(self, X):
        X_clean = X.fillna(0)
        if hasattr(X_clean, "values"):
            X_clean = X_clean.values
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_clean)
        preds = self.model.predict(X_clean)
        probs = np.clip(preds, 0, 1)
        return np.column_stack([1 - probs, probs])

    def get_config(self):
        return {"model_name": self.model_name, "params": self.kwargs}

    def __repr__(self):
        return f"{self.model_name}(type={self.model_name.replace('Model', '')})"

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


class RandomForestModel(BaseModel):
    """Random Forest Classifier - Versión CORREGIDA (Restrictiva)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parámetros por defecto RESTRICTIVOS (como el que SÍ funcionaba)
        default_params = {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_split": 30,
            "min_samples_leaf": 15,
            "max_features": 0.3,
            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(**default_params)
        self.kwargs = default_params

    def prefix_name(self) -> str:
        p = self.kwargs
        return (
            f"rf_n{p['n_estimators']}"
            f"_md{p['max_depth']}"
            f"_mss{p['min_samples_split']}"
            f"_msl{p['min_samples_leaf']}"
            f"_mf{p['max_features']}"
            f"_cwbal"
        )

    def feature_importances(self, feature_names):
        if hasattr(self.model, "feature_importances_"):
            return sorted(
                zip(feature_names, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        return []