# inference_rf.py
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Usa imports locales; si usas paquete, cámbialos por tus rutas de paquete.
from data_processing import read_test_data, read_csv

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent

MODELS_DIR = BASE_DIR / "trained_models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)

# Nombre EXACTO del modelo entrenado que compartiste
MODEL_FILENAME = "rf_n500_msl2_mfsqrt_cwbal_20251018_033701.pkl"


def load_model(model_filename: str):
    path = MODELS_DIR / model_filename
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    model = joblib.load(path)
    print(f"Modelo cargado de: {path}")
    if not hasattr(model, "predict_proba"):
        raise AttributeError("El objeto cargado no tiene método predict_proba().")
    return model


def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Reindexa X a las columnas usadas en fit (feature_names_in_) y
    llena faltantes con 0. También elimina columnas extra.
    """
    # Intentar obtener las columnas del estimador subyacente (RandomForest)
    feat_names = None
    if hasattr(model, "clf") and hasattr(model.clf, "feature_names_in_"):
        feat_names = list(model.clf.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)  # por si guardaste el RF directo

    if feat_names is None:
        raise AttributeError(
            "No se pudieron recuperar las columnas de entrenamiento (feature_names_in_). "
            "Asegúrate de haber entrenado con un DataFrame (no solo numpy array)."
        )

    # Debug útil
    missing = [c for c in feat_names if c not in X.columns]
    extra = [c for c in X.columns if c not in feat_names]
    if missing:
        print(f"[WARN] Faltan {len(missing)} columnas respecto a fit (se rellenarán con 0). Ejemplos: {missing[:10]}")
    if extra:
        print(f"[WARN] Hay {len(extra)} columnas no vistas en fit (se eliminarán). Ejemplos: {extra[:10]}")

    X_aligned = X.reindex(columns=feat_names, fill_value=0)

    # Asegurar tipo numérico
    for c in X_aligned.columns:
        if not np.issubdtype(X_aligned[c].dtype, np.number):
            X_aligned[c] = pd.to_numeric(X_aligned[c], errors="coerce").fillna(0)

    return X_aligned


def make_submission(model_filename: str = MODEL_FILENAME, outname: str | None = None):
    """
    Genera un archivo CSV con columnas:
      - purchase_id
      - probability
    con el mismo número de filas que el CSV de test.
    """
    # 1) Cargar modelo
    model = load_model(model_filename)

    # 2) Cargar test crudo (para purchase_id) y test procesado (para features)
    test_raw = read_csv("customer_purchases_test")   # contiene 'purchase_id'
    X_test = read_test_data()                        # features procesados

    # 2.1) Alinear columnas de X_test con lo visto en fit
    X_test = _align_features_to_model(X_test, model)

    # 3) Inferencia
    proba = model.predict_proba(X_test)
    proba = np.clip(proba, 0.0, 1.0)  # seguridad numérica

    # 4) Armar submission
    sub = pd.DataFrame({
        "purchase_id": test_raw["purchase_id"].values,
        "probability": proba
    })

    # 5) Guardar
    if outname is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outname = f"submission_{ts}.csv"
    out_path = SUBMISSIONS_DIR / outname
    sub.to_csv(out_path, index=False)
    print(f"Submission guardada en: {out_path.resolve()}")

    # 6) Chequeos de integridad
    assert len(sub) == len(test_raw), "El submission no tiene el mismo número de filas que el test."
    assert list(sub.columns) == ["purchase_id", "probability"], "El submission debe tener dos columnas exactas."

    return out_path


if __name__ == "__main__":
    make_submission()
