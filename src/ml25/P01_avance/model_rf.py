# model_rf.py
# Random Forest básico para Customer Purchases (sin calibración)

# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML
import numpy as np
from typing import Optional, Union, Dict, Any
from sklearn.ensemble import RandomForestClassifier


CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(
        self,
        n_estimators: int = 800,
        max_depth: Optional[int] = 25,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Inicializa un modelo RandomForestClassifier estándar.
        """
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.clf = RandomForestClassifier(**self.params)

    # -----------------------------------------------------
    # API PRINCIPAL
    # -----------------------------------------------------
    def fit(self, X, y):
        """
        Entrena el modelo Random Forest.
        """
        self.clf.fit(X, y)
        return self

    def predict(self, X, threshold: float = 0.5):
        """
        Predice etiquetas binarias según un umbral sobre predict_proba.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def predict_proba(self, X):
        """
        Devuelve SIEMPRE la probabilidad de la clase positiva (1),
        independientemente del orden de clases en clf.classes_.
        """
        if not hasattr(self.clf, "predict_proba"):
            raise AttributeError("Este clasificador no soporta predict_proba().")

        proba = self.clf.predict_proba(X)

        # Asegurar que tomamos la probabilidad de la clase 1
        if hasattr(self.clf, "classes_") and 1 in self.clf.classes_:
            idx = list(self.clf.classes_).index(1)
            return proba[:, idx]
        else:
            # fallback si no hay 1 en clases
            return proba[:, -1]

    # -----------------------------------------------------
    # UTILIDADES
    # -----------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Devuelve los hiperparámetros clave del modelo.
        """
        cfg = self.clf.get_params()
        return {
            "type": "RandomForestClassifier",
            "n_estimators": cfg.get("n_estimators"),
            "max_depth": cfg.get("max_depth"),
            "min_samples_split": cfg.get("min_samples_split"),
            "min_samples_leaf": cfg.get("min_samples_leaf"),
            "max_features": cfg.get("max_features"),
            "class_weight": cfg.get("class_weight"),
            "random_state": cfg.get("random_state"),
            "n_jobs": cfg.get("n_jobs"),
        }

    # -----------------------------------------------------
    # GUARDAR Y CARGAR
    # -----------------------------------------------------
    def save(self, prefix: str):
        """
        Guarda el modelo en trained_models/<prefix>_<timestamp>.pkl
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = MODELS_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)

        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        """
        Carga un modelo desde trained_models/<filename>
        """
        filepath = MODELS_DIR / filename
        model = joblib.load(filepath)
        print(f"{repr(self)} || Model loaded from {filepath}")
        return model

    # -----------------------------------------------------
    # REPRESENTACIÓN
    # -----------------------------------------------------
    def __repr__(self):
        cfg = self.clf.get_params()
        ne = cfg.get("n_estimators", "?")
        md = cfg.get("max_depth", "?")
        msl = cfg.get("min_samples_leaf", "?")
        mf = cfg.get("max_features", "?")
        cw = cfg.get("class_weight", "?")
        return f"RandomForest(n={ne}, depth={md}, leaf={msl}, max_feat={mf}, cw={cw})"
