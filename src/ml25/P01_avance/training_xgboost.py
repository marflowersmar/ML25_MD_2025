# training_xgboost.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path
from data_processing import build_training_table, preprocess
from model_xgboost import XGBoostModel, save_feature_importance


# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_SAVE_PATH = Path(__file__).parent / "models" / "xgboost_model.pkl"
FEATURE_IMPORTANCE_PATH = Path(__file__).parent / "models" / "feature_importance.csv"
MODELS_DIR = Path(__file__).parent / "models"


def main():
    print("=== ENTRENAMIENTO DEL MODELO XGBOOST MEJORADO ===")
    
    # Crear directorio de modelos si no existe
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Construir dataset de entrenamiento con mejor balance
    print("Construyendo tabla de entrenamiento...")
    X_raw, y = build_training_table(
        n_per_positive=5,  # Más negativos para mejor balance
        smart=True, 
        random_state=RANDOM_STATE
    )
    
    print(f"Dataset shape: {X_raw.shape}")
    print(f"Distribución de labels: {y.value_counts()}")
    print(f"Proporción positiva: {y.mean():.3f}")
    
    # 2. Split train/validation
    print("Dividiendo en train/validation...")
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    
    # 3. Preprocesamiento
    print("Preprocesando datos...")
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)
    
    print(f"X_train shape: {X_tr.shape}")
    print(f"X_val shape: {X_va.shape}")
    
    # 4. Calcular scale_pos_weight para balancear clases
    scale_pos_weight = len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # 5. Entrenar modelo con parámetros mejorados
    print("Entrenando modelo XGBoost...")
    model = XGBoostModel({
        'max_depth': 8,
        'learning_rate': 0.05,  # Learning rate más bajo para mejor convergencia
        'n_estimators': 800,   # Más árboles
        'objective': 'binary:logistic',
        'eval_metric': 'auc',  # Optimizar para AUC
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,      # Regularización L1
        'reg_lambda': 1.0,     # Regularización L2
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight  # Balancear clases
    })
    
    # Entrenar sin early stopping
    model.fit(X_tr, y_tr)
    
    # 6. Evaluar en validation
    print("\n=== EVALUACIÓN EN VALIDATION ===")
    results = model.evaluate(X_va, y_va)
    
    # 7. Ver distribución de probabilidades
    y_pred_proba = model.predict_proba(X_va)
    print(f"Rango de probabilidades en validation: {y_pred_proba.min():.4f} - {y_pred_proba.max():.4f}")
    
    # 8. Guardar modelo y feature importance
    print("Guardando modelo...")
    model.save_model(MODEL_SAVE_PATH)
    save_feature_importance(model.feature_importance, FEATURE_IMPORTANCE_PATH)
    
    print("¡Entrenamiento completado!")
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")
    print(f"Feature importance guardado en: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()