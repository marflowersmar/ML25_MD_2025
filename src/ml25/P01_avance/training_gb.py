import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from model_gb import PurchaseModel, evaluate_model
from utils import setup_logger
from data_processing import read_train_data

def run_training():
    logger = setup_logger("training_xgb")

    # 1. Leer datos
    X, y = read_train_data()
    logger.info(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} features")

    # 2. Verificar desbalance
    class_dist = y.value_counts(normalize=True)
    logger.info(f"Distribución de clases:\n{class_dist}")

    # 3. Separar en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Instanciar modelo con regularización
    model = PurchaseModel(
        n_estimators=100,            # Menos árboles para evitar sobreajuste
        max_depth=4,                 # Árboles más simples
        learning_rate=0.1,           # Tasa de aprendizaje moderada
        subsample=0.8,               # Evita que el modelo vea todo el dataset
        colsample_bytree=0.8,        # Reduce sobreajuste por columnas
        scale_pos_weight=class_dist[0] / class_dist[1],  # Ajuste por desbalance
    )

    # 5. Entrenamiento
    model.fit(X_train, y_train)
    logger.info("Modelo XGBoost entrenado")

    # 6. Evaluación
    train_roc = roc_auc_score(y_train, model.predict_proba(X_train))
    val_roc = roc_auc_score(y_val, model.predict_proba(X_val))
    logger.info(f"ROC-AUC entrenamiento: {train_roc:.4f}")
    logger.info(f"ROC-AUC validación: {val_roc:.4f}")

    metrics = evaluate_model(model, X_val, y_val, title="XGB")
    logger.info(f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

    # 7. Guardar modelo
    model.save("xgb")

if __name__ == "__main__":
    run_training()
