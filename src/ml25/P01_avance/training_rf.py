# training_rf.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from model_rf import PurchaseModel
from utils import setup_logger
from data_processing import build_training_table, preprocess


def run_training(
    test_size: float = 0.2,
    random_state: int = 42,
    classifier_name: str = "random_forest",
):
    """
    Entrena y evalúa un modelo Random Forest con split antes del preprocesamiento
    y preprocesador ajustado exclusivamente con el conjunto de entrenamiento.
    """

    logger = setup_logger(f"training_{classifier_name}")
    logger.info("Inicio del entrenamiento con Random Forest")

    # 1) Construir la tabla canónica cruda para entrenamiento (sin fuga)
    X_raw, y = build_training_table(n_per_positive=1, smart=True, random_state=random_state)
    logger.info(f"Dataset crudo: X={X_raw.shape}, y={y.shape}")

    # 2) Split estratificado antes de cualquier fit del preprocesador
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Split realizado -> X_train: {X_train_raw.shape}, X_val: {X_val_raw.shape}")

    # 3) Preprocesamiento (ajustar preprocessor solo con train)
    X_train = preprocess(X_train_raw, training=True)
    X_val = preprocess(X_val_raw, training=False)
    logger.info(f"Preprocesamiento completado -> X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 4) Instanciación del modelo
    model = PurchaseModel(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info(f"Configuración del modelo: {model.get_config()}")

    # 5) Entrenamiento
    model.fit(X_train, y_train)
    logger.info("Modelo entrenado correctamente")

    # 6) Evaluación
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    y_proba_train = model.predict_proba(X_train)
    y_proba_val = model.predict_proba(X_val)

    roc_train = roc_auc_score(y_train, y_proba_train)
    roc_val = roc_auc_score(y_val, y_proba_val)
    pr_val = average_precision_score(y_val, y_proba_val)
    f1_val = f1_score(y_val, y_pred_val)

    logger.info(f"ROC-AUC Entrenamiento: {roc_train:.4f}")
    logger.info(f"ROC-AUC Validación: {roc_val:.4f} | PR-AUC: {pr_val:.4f} | F1: {f1_val:.4f}")

    cm = confusion_matrix(y_val, y_pred_val)
    logger.info("\nMatriz de confusión (validación):\n" + str(cm))
    logger.info("\n" + classification_report(y_val, y_pred_val, digits=4))

    if len(np.unique(y_pred_val)) == 1:
        logger.warning("El modelo predijo una sola clase en validación. Revisa balance o features.")

    # 7) Guardar modelo
    prefix = (
        f"rf_n{model.clf.n_estimators}"
        f"_md{model.clf.max_depth}"
        f"_msl{model.clf.min_samples_leaf}"
        f"_mf{model.clf.max_features}"
        f"_cw{'bal' if model.clf.class_weight == 'balanced' else 'none'}"
    )
    model_path = model.save(prefix=prefix)
    logger.info(f"Modelo guardado en: {model_path}")

    # 8) Resultados finales
    results = {
        "roc_train": roc_train,
        "roc_val": roc_val,
        "pr_val": pr_val,
        "f1_val": f1_val,
        "model_path": model_path,
    }
    logger.info(f"Resultados finales: {results}")
    return results


if __name__ == "__main__":
    results = run_training()
    print("\nEntrenamiento completado con éxito.")
    print(results)
