import sys
import pandas as pd
import joblib
from pathlib import Path
from data_processing import read_test_data, preprocess
from model_gb import PurchaseModel
from utils import setup_logger

# ✅ Evita errores Unicode con emojis (opcional)
# Si quieres mantener emojis, descomenta esta línea:
# sys.stdout.reconfigure(encoding='utf-8')

CURRENT_FILE = Path(__file__).resolve()
MODEL_DIR = CURRENT_FILE.parent / "trained_models"
RESULTS_PATH = CURRENT_FILE.parent / "inference_results.csv"


def run_inference():
    logger = setup_logger("inference_xgb")

    # 1️⃣ Cargar modelo entrenado
    model_files = list(MODEL_DIR.glob("xgb_*.pkl"))
    if not model_files:
        logger.error("No se encontró ningún modelo entrenado. Ejecuta primero training_xgb.py.")
        return

    latest_model = sorted(model_files)[-1]
    logger.info(f"Cargando modelo: {latest_model.name}")
    model = joblib.load(latest_model)

    # 2️⃣ Leer y preprocesar datos de prueba
    logger.info("Cargando datos de prueba...")
    X_test = read_test_data()
    X_test_proc = preprocess(X_test, training=False)
    logger.info(f"Datos procesados: {X_test_proc.shape}")

    # 3️⃣ Realizar inferencia
    logger.info("Ejecutando predicciones...")
    preds = model.predict(X_test_proc)
    probs = model.predict_proba(X_test_proc)

    # 4️⃣ Guardar resultados
    results = pd.DataFrame({
        "customer_id": X_test.index,
        "prediction": preds,
        "probability": probs
    })
    results.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Resultados guardados en {RESULTS_PATH}")

    # 5️⃣ Mostrar muestra de salida
    logger.info("Ejemplo de resultados:")
    logger.info(f"\n{results.head(10)}")

    return results


if __name__ == "__main__":
    run_inference()
