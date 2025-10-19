import argparse
import os
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from model_rf import PurchaseModel
from data_processing import read_test_data
from sklearn.utils.validation import check_is_fitted



def find_latest_model(models_dir: Path) -> Path:
    """Busca el modelo .pkl más reciente en la carpeta indicada."""
    pkl_files = list(models_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No se encontraron modelos .pkl en {models_dir}")
    latest = max(pkl_files, key=os.path.getmtime)
    print(f"📦 Modelo más reciente encontrado: {latest.name}")
    return latest


def load_model(model_path: str | Path) -> PurchaseModel:
    """
    Carga un modelo entrenado de forma robusta.
    Prioriza joblib.load (trae el estado entrenado). Si lo que retorna es:
      - PurchaseModel: lo usa directo.
      - Un estimador (RandomForestClassifier): lo envuelve en PurchaseModel.
    Como último recurso intenta el método .load del wrapper (si existiera).
    """
    model_path = Path(model_path)

    # 1) Intento principal: joblib.load (recupera estado entrenado)
    try:
        obj = joblib.load(model_path)
        # Puede ser el wrapper completo o el estimador crudo:
        if isinstance(obj, PurchaseModel):
            pm = obj
        else:
            pm = PurchaseModel()
            pm.clf = obj
        # Verificar que esté fitted
        try:
            check_is_fitted(pm.clf)
        except Exception as e:
            raise RuntimeError(
                f"El estimador cargado no está entrenado (no-fitted). "
                f"Reentrena o verifica cómo se guardó el modelo: {e}"
            )
        return pm
    except Exception:
        pass  # caemos al intento 2

    # 2) Intento secundario: método .load del wrapper (si lo hubiera)
    try:
        pm = PurchaseModel()
        if hasattr(pm, "load") and callable(getattr(pm, "load")):
            pm.load(str(model_path))  # método de instancia
            # Checar fitted
            check_is_fitted(pm.clf)
            return pm
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo desde {model_path}: {e}")

def run_inference(model_path: Path, threshold: float = 0.5, output_path: Path = None):
    """Ejecuta inferencia con el modelo Random Forest."""
    print(f"Cargando modelo desde: {model_path}")
    model = load_model(model_path)

    print("Cargando datos de test...")
    X_test = read_test_data()

    # Mantener el mismo orden que el dataset original
    purchase_ids = pd.Series(range(len(X_test)), name="ID")
    X_input = X_test.copy()

    print(f"Aplicando predicciones con threshold={threshold}...")
    y_proba = model.predict_proba(X_input)
    y_pred = (y_proba >= threshold).astype(int)

    # Solo guardamos ID y predicción
    results = pd.DataFrame({
        "ID": purchase_ids,
        "prediction": y_pred
    })

    # Guardar resultados
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = model_path.parent / f"inference_results_{timestamp}.csv"

    results.to_csv(output_path, index=False)
    print(f"✅ Resultados guardados en: {output_path}")
    print(results.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia con modelo Random Forest")
    parser.add_argument("--model", type=str, help="Ruta del modelo .pkl (opcional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de decisión (default=0.5)")
    parser.add_argument("--output", type=str, help="Archivo CSV de salida (opcional)")
    args = parser.parse_args()

    # Determinar ruta del modelo
    if args.model:
        model_path = Path(args.model)
    else:
        models_dir = Path(__file__).resolve().parent / "trained_models"
        model_path = find_latest_model(models_dir)

    output_path = Path(args.output) if args.output else None
    run_inference(model_path=model_path, threshold=args.threshold, output_path=output_path)
