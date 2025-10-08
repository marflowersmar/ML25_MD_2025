# models.py
from pathlib import Path
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

# Data management
CURRENT_DIR = Path.cwd()
MODELS_DIR = CURRENT_DIR / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

class GradientBoostingModel:
    def __init__(self, **params):
        self.model_type = "GradientBoosting"
        # Parámetros ESPECÍFICOS de Gradient Boosting
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,      # ← CRÍTICO en GB
            'max_depth': 6,            # ← Más shallow en GB
            'subsample': 0.8,          # ← Stochastic GB
            'random_state': 42
            # NO tiene n_jobs (GB es secuencial)
        }
        self.params = {**default_params, **params}
        self.model = self._build_model()
        self.is_fitted = False
        
    def _build_model(self):
        """Construye el modelo Gradient Boosting"""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(**self.params)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            print("INFO: Usando GradientBoosting de sklearn")
            return GradientBoostingClassifier(**self.params)

    def fit(self, X, y):
        """Entrenamiento del modelo"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        return self.model.predict_proba(X)

    def get_config(self):
        return {
            'model_type': self.model_type,
            'hyperparameters': self.params,
            'is_fitted': self.is_fitted
        }

    def calculate_precision_at_k(self, y_true, y_proba, k=100):
        """Calcula precision@k"""
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_proba[:, 1])
        k = max(1, min(k, len(y_scores)))
        top_k_idx = np.argsort(-y_scores)[:k]
        return float(np.mean(y_true[top_k_idx]))

    def evaluate(self, X_test, y_test):
        """Evaluación completa con métricas"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
            
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else float("nan"),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'precision_at_100': self.calculate_precision_at_k(y_test, y_proba, 100)
        }
        return metrics

    def save(self, prefix: str):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GB_{prefix}_{now}.pkl"
        filepath = MODELS_DIR / filename
        joblib.dump(self.model, filepath)
        print(f"GradientBoosting model saved to {filepath}")
        return filepath

    def __repr__(self):
        return f"GradientBoostingModel(fitted={self.is_fitted})"


class RandomForestModel:
    def __init__(self, **params):
        self.model_type = "RandomForest"
        # Parámetros ESPECÍFICOS de Random Forest
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,           # ← Más profundo en RF
            'min_samples_split': 2,    # ← Típico en RF  
            'min_samples_leaf': 1,     # ← Típico en RF
            'bootstrap': True,         # ← BAGGING (característica de RF)
            'random_state': 42,
            'n_jobs': -1               # ← Paralelización (RF sí puede)
        }
        self.params = {**default_params, **params}
        self.model = self._build_model()
        self.is_fitted = False
        
    def _build_model(self):
        """Construye el modelo Random Forest"""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**self.params)

    def fit(self, X, y):
        """Entrenamiento del modelo"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        return self.model.predict_proba(X)

    def get_config(self):
        return {
            'model_type': self.model_type,
            'hyperparameters': self.params,
            'is_fitted': self.is_fitted
        }

    def calculate_precision_at_k(self, y_true, y_proba, k=100):
        """Calcula precision@k"""
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_proba[:, 1])
        k = max(1, min(k, len(y_scores)))
        top_k_idx = np.argsort(-y_scores)[:k]
        return float(np.mean(y_true[top_k_idx]))

    def evaluate(self, X_test, y_test):
        """Evaluación completa con métricas"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
            
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else float("nan"),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'precision_at_100': self.calculate_precision_at_k(y_test, y_proba, 100)
        }
        return metrics

    def save(self, prefix: str):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RF_{prefix}_{now}.pkl"
        filepath = MODELS_DIR / filename
        joblib.dump(self.model, filepath)
        print(f"RandomForest model saved to {filepath}")
        return filepath

    def __repr__(self):
        return f"RandomForestModel(fitted={self.is_fitted})"


# Solo para testing si se ejecuta directamente
if __name__ == "__main__":
    print("GradientBoostingModel listo!")
    print("RandomForestModel listo!")