# ML
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Modelo Random Forest personalizado
from model_rf import PurchaseModel  # 👈 aquí usas tu archivo de modelo

# Utilidades
from utils import setup_logger
from data_processing import read_train_data


def run_training(X, y, classifier_name: str = "random_forest", test_size: float = 0.2, random_state: int = 42):
    logger = setup_logger(f"training_{classifier_name}")
    logger.info("Inicio del entrenamiento con Random Forest")

    # 1️⃣ División en entrenamiento y validación
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Shapes → X_train: {X_tr.shape}, X_val: {X_va.shape}")

    # 2️⃣ Instanciación del modelo
    model = PurchaseModel(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info(f"Configuración del modelo: {model.get_config()}")

    # 3️⃣ Entrenamiento
    model.fit(X_tr, y_tr)
    logger.info("✅ Modelo entrenado correctamente")

    # 4️⃣ Validación
    proba_va = model.predict_proba(X_va)
    pred_va = (proba_va >= 0.5).astype(int)

    roc = roc_auc_score(y_va, proba_va)
    pr = average_precision_score(y_va, proba_va)
    logger.info(f"[VALIDACIÓN] ROC-AUC={roc:.5f} | PR-AUC={pr:.5f}")
    logger.info("\n" + classification_report(y_va, pred_va, digits=4))

    # 5️⃣ Guardar modelo
    prefix = (
        f"rf_n{model.clf.n_estimators}"
        f"_msl{model.clf.min_samples_leaf}"
        f"_mf{model.clf.max_features}"
        f"_cw{'bal' if model.clf.class_weight == 'balanced' else 'none'}"
    )
    path = model.save(prefix=prefix)
    logger.info(f"📁 Modelo guardado en: {path}")

    return {"model": model, "val_roc_auc": roc, "val_pr_auc": pr, "model_path": path}


if __name__ == "__main__":
    # Cargar datos ya procesados
    X, y = read_train_data()

    # Ejecutar entrenamiento
    results = run_training(X, y, classifier_name="random_forest")
    print("\nResultados finales:")
    print(results)
