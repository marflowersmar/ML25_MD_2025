import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Usa imports locales
from data_processing import read_test_data, read_csv

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent

MODELS_DIR = BASE_DIR / "trained_models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)

# Nombre del modelo XGBoost previamente guardado
MODEL_FILENAME = "xgb_20251019_103052.pkl"  


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
    feat_names = None
    if hasattr(model, "clf") and hasattr(model.clf, "feature_names_in_"):
        feat_names = list(model.clf.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)

    if feat_names is None:
        raise AttributeError(
            "No se pudieron recuperar las columnas de entrenamiento (feature_names_in_). "
            "Asegúrate de haber entrenado con un DataFrame (no solo numpy array)."
        )

    missing = [c for c in feat_names if c not in X.columns]
    extra = [c for c in X.columns if c not in feat_names]
    if missing:
        print(f"[WARN] Faltan {len(missing)} columnas respecto a fit (se rellenarán con 0). Ejemplos: {missing[:10]}")
    if extra:
        print(f"[WARN] Hay {len(extra)} columnas no vistas en fit (se eliminarán). Ejemplos: {extra[:10]}")

    X_aligned = X.reindex(columns=feat_names, fill_value=0)

    for c in X_aligned.columns:
        if not np.issubdtype(X_aligned[c].dtype, np.number):
            X_aligned[c] = pd.to_numeric(X_aligned[c], errors="coerce").fillna(0)

    return X_aligned


def make_submission(model_filename: str = MODEL_FILENAME, outname: str | None = None):
    model = load_model(model_filename)

    test_raw = read_csv("customer_purchases_test")
    X_test = read_test_data()

    X_test = _align_features_to_model(X_test, model)

    proba = model.predict_proba(X_test)
    proba = np.clip(proba, 0.0, 1.0)

    sub = pd.DataFrame({
        "purchase_id": test_raw["purchase_id"].values,
        "probability": proba
    })

    if outname is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outname = f"submission_xgb_{ts}.csv"
    out_path = SUBMISSIONS_DIR / outname
    sub.to_csv(out_path, index=False)
    print(f"Submission guardada en: {out_path.resolve()}")

    assert len(sub) == len(test_raw), "El submission no tiene el mismo número de filas que el test."
    assert list(sub.columns) == ["purchase_id", "probability"], "El submission debe tener dos columnas exactas."

    return out_path


if __name__ == "__main__":
    make_submission()
