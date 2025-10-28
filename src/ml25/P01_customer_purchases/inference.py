import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from ml25.P01_customer_purchases.boilerplate.data_processing import read_test_data

# === Configuración de rutas ===
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
RESULTS_DIR = BASE_DIR / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR = BASE_DIR / "trained_models"

# === Cargar modelo ===
def load_model(filename: str):
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {filepath}")
    model = joblib.load(filepath)
    print(f"✅ Modelo cargado desde: {filepath}")
    return model

# === Inferencia ===
def run_inference(model_name: str, X: pd.DataFrame) -> pd.DataFrame:
    model = load_model(model_name)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else np.zeros(len(X))
    results = pd.DataFrame({
        "ID": X.index,
        "prediction": preds,
        "probability": probs
    })
    return results

# === Curva ROC ===
def plot_roc(y_true, y_proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    # Cargar datos de test
    X_test = read_test_data()

    # Nombre del modelo entrenado
    model_name = "xgb_20251019_103052.pkl"  # actualiza si es necesario

    # Ejecutar inferencia
    results = run_inference(model_name, X_test)

    # Guardar resultados
    out_path = RESULTS_DIR / f"submission_xgb.csv"
    results.to_csv(out_path, index=False)
    print(f"📝 Submission guardada en: {out_path.resolve()}")

