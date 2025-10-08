# ML
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split

# Custom
from utils import setup_logger
from data_processing import read_train_data
from models import GradientBoostingModel, RandomForestModel


def run_training(X, y, classifier: str, model_type: str = "gb"):
    logger = setup_logger(f"training_{classifier}")
    logger.info("Iniciando entrenamiento...")
    
    # 1. Separar en entrenamiento y validacion
    logger.info("Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. Entrenamiento del modelo
    logger.info(f"Creando modelo {model_type.upper()}...")
    
    if model_type.lower() == "gb":
        model = GradientBoostingModel(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
    elif model_type.lower() == "rf":
        model = RandomForestModel(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("model_type debe ser 'gb' o 'rf'")
    
    logger.info(f"Modelo creado: {model.get_config()}")
    
    # Entrenar modelo
    logger.info("Entrenando modelo...")
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado")
    
    # 3. Validacion
    logger.info("Evaluando modelo...")
    
    # Usar el método evaluate del modelo
    metrics = model.evaluate(X_test, y_test)
    
    # Classification report detallado
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info("Metricas obtenidas:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    if '1' in class_report:
        logger.info("Classification Report:")
        logger.info(f"  Precision: {class_report['1']['precision']:.4f}")
        logger.info(f"  Recall: {class_report['1']['recall']:.4f}")
        logger.info(f"  F1-Score: {class_report['1']['f1-score']:.4f}")
    
    # 4. Guardar modelo
    logger.info("Guardando modelo...")
    saved_path = model.save(classifier)
    logger.info(f"Modelo guardado en: {saved_path}")
    
     # 5. Registrar métricas en CSV
    import pandas as pd
    import os
    from datetime import datetime

    log_row = pd.DataFrame([{
        "timestamp": str(datetime.now()),
        "classifier": classifier,
        "model_type": model.get_config()['model_type'],
        **model.get_config()['hyperparameters'],
        **metrics
    }])

    log_file = "experiment_log.csv"
    if os.path.exists(log_file):
        log_row.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_row.to_csv(log_file, index=False)

    logger.info("📊 Métricas registradas en experiment_log.csv")

    return {
        'model': model,
        'metrics': metrics,
        'classification_report': class_report,
        'saved_path': saved_path
    }    

def run_all_models(X, y):
    """
    Ejecuta entrenamiento para ambos modelos
    """
    results = {}
    
    logger_main = setup_logger("training_main")
    logger_main.info("Iniciando entrenamiento de todos los modelos...")
    
    logger_main.info("=" * 50)
    logger_main.info("ENTRENANDO GRADIENT BOOSTING")
    logger_main.info("=" * 50)
    results['gradient_boosting'] = run_training(
        X, y, 
        classifier="gradient_boosting",
        model_type="gb"
    )
    
    logger_main.info("=" * 50)
    logger_main.info("ENTRENANDO RANDOM FOREST")
    logger_main.info("=" * 50)
    results['random_forest'] = run_training(
        X, y, 
        classifier="random_forest", 
        model_type="rf"
    )
    
    # Comparar resultados
    logger_main.info("=" * 50)
    logger_main.info("COMPARACION DE MODELOS")
    logger_main.info("=" * 50)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        logger_main.info(f"{model_name.upper()}:")
        for metric, value in metrics.items():
            logger_main.info(f"  {metric}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    # Cargar datos
    X, y = read_train_data()
    
    # Ejecutar ambos modelos
    results = run_all_models(X, y)
    
    print("Entrenamiento completado!")
    print("Los modelos se guardaron en la carpeta 'trained_models/'")