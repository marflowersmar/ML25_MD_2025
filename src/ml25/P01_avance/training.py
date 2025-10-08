# training.py
# Entrenamiento del modelo + evaluación + tracking
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import json
import logging
import sys
import os

# Agregar el directorio actual al path para imports locales
sys.path.append(os.path.dirname(__file__))

# Importar módulos locales
from data_processing import read_train_data
from model import GradientBoostingPurchaseModel


def setup_logger(name: str, log_dir: str = "./logs"):
    """
    Setup a logger that writes to console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def calculate_precision_at_k(y_true, y_proba, k=5):
    """
    Calcula precision@k - métrica para sistemas de recomendación
    """
    if len(y_true) < k:
        k = len(y_true)
    
    sorted_indices = np.argsort(y_proba[:, 1])[::-1]
    top_k_indices = sorted_indices[:k]
    
    true_positives = np.sum(y_true.iloc[top_k_indices] == 1)
    precision = true_positives / k
    
    return precision


def calculate_all_metrics(model, X_test, y_test):
    """
    Calcula todas las métricas requeridas para el experimento
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba[:, 1]),
        'precision_at_5': calculate_precision_at_k(y_test, y_proba, k=5),
        'precision_at_10': calculate_precision_at_k(y_test, y_proba, k=10),
    }
    
    return metrics


def log_experiment_results(experiment_name, params, metrics, feature_importance, model_path):
    """
    Log estructurado de resultados del experimento
    """
    logger = setup_logger("experiment_tracker")
    
    experiment_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'metrics': metrics,
        'model_path': str(model_path)
    }
    
    if feature_importance is not None:
        experiment_data['top_features'] = feature_importance.head(10).to_dict('records')
    
    # Log en archivo JSON
    log_file = f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    # Log en consola
    logger.info(f"EXPERIMENTO: {experiment_name}")
    logger.info(f"Parametros: {params}")
    logger.info(f"Metricas - Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Precision@5: {metrics['precision_at_5']:.4f}, Precision@10: {metrics['precision_at_10']:.4f}")
    
    if feature_importance is not None:
        top_5 = feature_importance.head(5)['feature'].tolist()
        logger.info(f"Top 5 features: {top_5}")
    
    logger.info(f"Modelo guardado: {model_path}")
    
    return experiment_data


def run_training(X, y, experiment_name="GradientBoosting_Experiment"):
    """
    Entrenamiento completo con tracking de experimentos
    """
    logger = setup_logger(f"training_{experiment_name}")
    logger.info("Iniciando entrenamiento Gradient Boosting")
    
    # Verificar dimensiones de los datos
    logger.info(f"Dimensiones - X: {X.shape}, y: {y.shape}")
    
    if X.shape[0] != y.shape[0]:
        logger.error(f"ERROR: Dimensiones inconsistentes. X tiene {X.shape[0]} muestras, y tiene {y.shape[0]}")
        raise ValueError("X e y deben tener el mismo número de muestras")
    
    # 1. Separar en entrenamiento y validación
    logger.info("Dividiendo datos en train/validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}")
    logger.info(f"Distribucion labels - Train: {y_train.value_counts().to_dict()}")
    
    # 2. Configuración del modelo
    gb_params = {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'random_state': 42,
        'dataset_version': 'v1.0_negative_gen'
    }
    
    # 3. Entrenamiento del modelo
    model = GradientBoostingPurchaseModel(**gb_params)
    
    logger.info("Entrenando modelo Gradient Boosting")
    model.fit(X_train, y_train, X_val, y_val)
    
    # 4. Validación y métricas
    logger.info("Calculando metricas")
    metrics = calculate_all_metrics(model, X_val, y_val)
    
    # 5. Feature importance
    feature_importance = model.get_feature_importance(X_train.columns.tolist())
    
    # 6. Guardar modelo
    logger.info("Guardando modelo")
    prefix = f"GB_{gb_params['n_estimators']}est_{gb_params['max_depth']}depth"
    model_path = model.save(prefix)
    
    # 7. Log de resultados del experimento
    experiment_data = log_experiment_results(
        experiment_name, gb_params, metrics, feature_importance, model_path
    )
    
    return model, metrics, feature_importance, experiment_data


def run_ablacion_study(X, y):
    """
    Ejecuta múltiples experimentos para comparar configuraciones
    """
    logger = setup_logger("ablacion_study")
    logger.info("Iniciando estudio de ablacion")
    
    configs = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4},
        {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 6},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8},
    ]
    
    results = []
    for i, config in enumerate(configs):
        logger.info(f"Ejecutando configuracion {i+1}/{len(configs)}: {config}")
        
        experiment_name = f"GB_Ablacion_{i+1}"
        model, metrics, feature_importance, experiment_data = run_training(
            X, y, experiment_name=experiment_name
        )
        
        results.append({
            'config': config,
            'metrics': metrics,
            'model': model,
            'experiment_data': experiment_data
        })
    
    # Comparar resultados
    logger.info("Resumen estudio ablacion:")
    best_roc_auc = 0
    best_config = None
    
    for i, result in enumerate(results):
        roc_auc = result['metrics']['roc_auc']
        accuracy = result['metrics']['accuracy']
        logger.info(f"Config {i+1}: {result['config']}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_config = result['config']
    
    logger.info(f"Mejor configuracion: {best_config} con ROC AUC: {best_roc_auc:.4f}")
    
    return results


if __name__ == "__main__":
    try:
        # Cargar datos procesados (ahora del dataset corregido)
        X, y = read_train_data()
        print(f"Datos cargados: {X.shape}, Labels: {y.shape}")
        print(f"Distribucion de labels: {y.value_counts()}")
        
        # Ejecutar entrenamiento principal
        model, metrics, feature_importance, experiment_data = run_training(X, y)
        
        # Ejecutar estudio de ablación (opcional)
        # results = run_ablacion_study(X, y)
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()