# model_xgboost.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from pathlib import Path


class XGBoostModel:
    def __init__(self, model_params=None):
        if model_params is None:
            model_params = {
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 800,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        self.model_params = model_params
        self.model = None
        self.feature_importance = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena el modelo XGBoost"""
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict_proba(self, X):
        """Predice probabilidades para clase positiva"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=0.5):
        """Predice clases binarias"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def evaluate(self, X, y, threshold=0.5):
        """Evalúa el modelo con métricas"""
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print("Classification Report:")
        print(classification_report(y, y_pred))
        
        auc = roc_auc_score(y, y_pred_proba)
        print(f"AUC Score: {auc:.4f}")
        
        return {
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        joblib.dump(self.model, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo pre-entrenado"""
        self.model = joblib.load(filepath)
        print(f"Modelo cargado desde: {filepath}")
        return self


def save_feature_importance(feature_importance_df, filepath):
    """Guarda la importancia de features"""
    feature_importance_df.to_csv(filepath, index=False)
    print(f"Feature importance guardado en: {filepath}")