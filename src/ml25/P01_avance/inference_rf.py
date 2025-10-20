# inference_rf.py
import argparse
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.utils.validation import check_is_fitted

from model_rf import PurchaseModel
from data_processing import read_test_data

def find_latest_model(models_dir: Path) -> Path:
    """Encuentra el modelo más reciente"""
    files = list(models_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No se encontraron modelos en {models_dir}")
    latest = max(files, key=os.path.getmtime)
    print(f"📦 Modelo: {latest.name}")
    return latest

def load_model(model_path: str | Path) -> PurchaseModel:
    """Carga el modelo"""
    obj = joblib.load(model_path)
    if isinstance(obj, PurchaseModel):
        pm = obj
    else:
        pm = PurchaseModel()
        pm.clf = obj
    check_is_fitted(pm.clf)
    print(f"✅ Modelo cargado: {pm._safe_repr()}")
    return pm

def load_optimal_threshold(models_dir: Path) -> float:
    """Carga el threshold óptimo"""
    opt_file = models_dir / "optimal_threshold.txt"
    if opt_file.exists():
        th = float(opt_file.read_text().strip())
        print(f"🎯 Threshold óptimo: {th:.4f}")
        return th
    else:
        print("⚠️  Usando threshold por defecto: 0.5")
        return 0.5

def run_inference():
    """Ejecuta inferencia automática"""
    print("🚀 INICIANDO INFERENCIA AUTOMÁTICA")
    
    # Configuración
    models_dir = Path(__file__).resolve().parent / "trained_models"
    model_path = find_latest_model(models_dir)
    threshold = load_optimal_threshold(models_dir)
    
    # Cargar modelo
    model = load_model(model_path)
    
    # Cargar datos
    print("📊 Cargando datos de test...")
    X_test = read_test_data()
    print(f"📈 Dimensiones: {X_test.shape}")
    
    # Predecir
    print(f"🔮 Prediciendo con threshold={threshold:.4f}...")
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba >= threshold).astype(int)
    
    # Análisis de resultados
    n_total = len(y_pred)
    n_positive = y_pred.sum()
    n_negative = n_total - n_positive
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Total predicciones: {n_total}")
    print(f"   Clase 1 (Compra): {n_positive} ({n_positive/n_total:.1%})")
    print(f"   Clase 0 (No compra): {n_negative} ({n_negative/n_total:.1%})")
    
    # Estadísticas de probabilidades
    print(f"\n📈 ESTADÍSTICAS DE PROBABILIDAD:")
    print(f"   Mínimo: {y_proba.min():.4f}")
    print(f"   Máximo: {y_proba.max():.4f}") 
    print(f"   Media: {y_proba.mean():.4f}")
    print(f"   Mediana: {np.median(y_proba):.4f}")
    
    # Guardar resultados
    results = pd.DataFrame({
        "ID": range(n_total),
        "prediction": y_pred
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = models_dir / f"submission_{timestamp}.csv"
    results.to_csv(output_path, index=False)
    
    print(f"\n💾 Resultados guardados: {output_path}")
    print("👀 Primeras 10 predicciones:")
    print(results.head(10).to_string(index=False))
    
    return results

if __name__ == "__main__":
    # Siempre ejecutar en modo automático
    results = run_inference()
    print(f"\n🎉 INFERENCIA COMPLETADA - {len(results)} predicciones generadas")