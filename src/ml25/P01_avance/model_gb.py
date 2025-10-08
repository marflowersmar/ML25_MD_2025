# model_gb.py
# Modelo Gradient Boosting + tracking ligero de experimentos
from pathlib import Path
import joblib
from datetime import datetime
import os
import pandas as pd

# Data management
CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class GradientBoostingPurchaseModel:
    def __init__(self, dataset_version="v1.0", **params):
        self.model_type = "GradientBoosting"
        self.params = params
        self.dataset_version = dataset_version
        self.model = self._build_model()
        self.training_date = datetime.now()
        self.feature_importance_ = None
        self.is_fitted = False
        
    def _build_model(self):
        """Construye el modelo Gradient Boosting"""
        # Intentar XGBoost primero (mejor performance)
        try:
            from xgboost import XGBClassifier
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            final_params = {**default_params, **self.params}
            return XGBClassifier(**final_params)
        except ImportError:
            # Fallback a sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier
            print("INFO: Usando GradientBoosting de sklearn (instala xgboost para mejor performance)")
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            final_params = {**default_params, **self.params}
            return GradientBoostingClassifier(**final_params)

    def fit(self, X, y, X_val=None, y_val=None):
        """Entrenamiento con early stopping si hay validation set"""
        if X_val is not None and y_val is not None and hasattr(self.model, 'fit'):
            try:
                # Early stopping para XGBoost - parámetro correcto
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,  # CORREGIDO: 'rounds' no 'round'
                    verbose=False
                )
            except TypeError:
                # Si falla, usar entrenamiento normal
                print("Usando entrenamiento sin early stopping")
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)
        
        # Guardar importancia de features
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        return self.model.predict_proba(X)

    def get_config(self):
        """Return key hyperparameters for logging"""
        config = {
            'model_type': self.model_type,
            'dataset_version': self.dataset_version,
            'hyperparameters': self.params,
            'training_date': self.training_date.strftime("%Y-%m-%d %H:%M:%S"),
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            config['n_features'] = getattr(self.model, 'n_features_in_', None)
            # Para XGBoost
            if hasattr(self.model, 'best_iteration'):
                config['best_iteration'] = self.model.best_iteration
        
        return config

    def get_feature_importance(self, feature_names=None):
        """Obtener importancia de features para análisis"""
        if self.feature_importance_ is None:
            return None
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def save(self, prefix: str):
        """
        Save the model to disk in MODELS_DIR with filename:
        <prefix>_<timestamp>.pkl
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GB_{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)

        joblib.dump(self, filepath)
        print(f"GradientBoosting model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        """
        Load the model from MODELS_DIR/filename
        """
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"GradientBoosting model loaded from {filepath}")
        return model

    def __repr__(self):
        return f"GradientBoostingPurchaseModel(fitted={self.is_fitted})"


# PARA PROBAR QUE FUNCIONE
if __name__ == "__main__":
    print("GradientBoosting model_gb.py cargado correctamente!")
    
    # Probar creación del modelo
    model = GradientBoostingPurchaseModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    print(f"Modelo creado: {model}")
    print(f"Config: {model.get_config()}")