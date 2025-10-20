# diagnostic.py
import pandas as pd
import numpy as np
from data_processing import build_training_table, preprocess
from model_xgboost import XGBoostModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def main():
    print("=== DIAGNÓSTICO DEL MODELO ===")
    
    # 1. Cargar datos
    X_raw, y = build_training_table(n_per_positive=3, smart=True, random_state=42)
    print(f"Dataset shape: {X_raw.shape}")
    print(f"Distribución de labels: {y.value_counts()}")
    
    # 2. Split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Preprocesar
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)
    
    # 4. Entrenar modelo con más iteraciones
    model = XGBoostModel({
        'max_depth': 6,
        'learning_rate': 0.05,  # Más bajo para mejor convergencia
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'scale_pos_weight': len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
    })
    
    model.fit(X_tr, y_tr)
    
    # 5. Evaluar en validation
    print("\n=== EVALUACIÓN EN VALIDATION ===")
    y_pred_proba = model.predict_proba(X_va)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_va, y_pred))
    
    auc = roc_auc_score(y_va, y_pred_proba)
    print(f"AUC Score: {auc:.4f}")
    
    # 6. Ver distribución de predicciones en test
    print("\n=== PREDICCIONES EN TEST ===")
    from data_processing import read_test_data, read_csv
    
    X_test = read_test_data()
    test_predictions = model.predict(X_test)
    test_probas = model.predict_proba(X_test)
    
    print(f"Distribución en test: {pd.Series(test_predictions).value_counts().sort_index()}")
    print(f"Rango de probabilidades en test: {test_probas.min():.4f} - {test_probas.max():.4f}")
    print(f"Media de probabilidades: {test_probas.mean():.4f}")


if __name__ == "__main__":
    main()