# inference_xgboosting.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from data_processing import read_test_data, read_csv
from model_xgboost import XGBoostModel


# Configuración
MODEL_PATH = Path(__file__).parent / "models" / "xgboost_model.pkl"
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"


def main():
    print("=== INFERENCIA PARA KAGGLE ===")
    
    # 1. Cargar modelo
    print("Cargando modelo...")
    model = XGBoostModel()
    model.load_model(MODEL_PATH)
    
    # 2. Cargar y preprocesar datos de test
    print("Procesando datos de test...")
    X_test_processed = read_test_data()
    
    # 3. Leer el test original para obtener los purchase_id
    test_df_original = read_csv("customer_purchases_test")
    
    # 4. Hacer predicciones binarias (0 o 1)
    print("Haciendo predicciones binarias...")
    predictions = model.predict(X_test_processed)  # Usar predict() para 0/1
    
    # 5. Crear submission file con nombres exactos
    print("Creando archivo de submission...")
    submission_df = pd.DataFrame({
        'ID': test_df_original['purchase_id'],
        'prediction': predictions.astype(int)  # Asegurar que sean 0 o 1
    })
    
    # Verificar que tenga el mismo número de filas que el test original
    assert len(submission_df) == len(test_df_original), \
        f"El submission tiene {len(submission_df)} filas pero el test tiene {len(test_df_original)}"
    
    # 6. Guardar submission
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission guardado en: {SUBMISSION_PATH}")
    print(f"Shape del submission: {submission_df.shape}")
    print(f"Columnas: {submission_df.columns.tolist()}")
    
    # 7. Mostrar distribución de predicciones
    pred_counts = submission_df['prediction'].value_counts().sort_index()
    print(f"\nDistribución de predicciones:")
    for pred, count in pred_counts.items():
        print(f"  {pred}: {count} ({count/len(submission_df)*100:.1f}%)")
    
    print(f"\nPrimeras 10 filas del submission:")
    print(submission_df.head(10))


if __name__ == "__main__":
    main()