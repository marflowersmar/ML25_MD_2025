# training_gb.py

# ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from datetime import datetime

# Custom
from ml25.P01_customer_purchases.boilerplate.utils import setup_logger
from ml25.P01_customer_purchases.boilerplate.data_processing import read_train_data


def run_training(X, y, classifier: str):
    logger = setup_logger(f"training_{classifier}")
    logger.info("Inicio del entrenamiento con Gradient Boosting")

    # 1. Separar en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # 2. Entrenamiento del modelo
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Modelo entrenado correctamente")

    # 3. Validación
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, y_proba)
    pr = average_precision_score(y_val, y_proba)
    f1 = f1_score(y_val, y_pred)

    logger.info(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1-Score: {f1:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_val, y_pred, digits=4))

    # 4. Guardar modelo
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(__file__).resolve().parent / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{classifier}_{now}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modelo guardado en: {model_path}")

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "f1_score": f1,
        "model_path": str(model_path)
    }


if __name__ == "__main__":
    X, y = read_train_data()
    results = run_training(X, y, classifier="gradient_boosting")
    print("Entrenamiento finalizado con éxito.")
    print(results)
