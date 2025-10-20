# find_optimal_threshold.py
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

CURRENT_FILE = Path(__file__).resolve()
sys.path.append(str(CURRENT_FILE.parent.parent))

from data_processing import read_test_data, read_csv
from model_rf import RandomForestModel

MODELS_DIR = CURRENT_FILE.parent.parent / "trained_models"
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def find_optimal_threshold():
    """Encuentra automáticamente el mejor threshold para balance 30-50%"""
    
    print("=" * 70)
    print("🎯 BUSCADOR DE THRESHOLD ÓPTIMO")
    print("=" * 70)
    
    # Cargar modelo más reciente
    models = sorted(MODELS_DIR.glob("rf_*.pkl"), key=lambda p: p.stat().st_mtime)
    if not models:
        print("❌ No hay modelos disponibles")
        return
    
    latest_model = models[-1]
    print(f"📦 Modelo: {latest_model.name}")
    
    # Cargar modelo y datos
    model = joblib.load(latest_model)
    X_test = read_test_data()
    test_raw = read_csv("customer_purchases_test")
    ids = test_raw["purchase_id"].reset_index(drop=True)
    
    # Obtener probabilidades
    print("🔮 Calculando probabilidades...")
    proba = model.predict_proba(X_test)[:, 1]
    
    print(f"📊 Estadísticas de probabilidades:")
    print(f"   Mínimo: {proba.min():.3f}")
    print(f"   Máximo: {proba.max():.3f}") 
    print(f"   Media: {proba.mean():.3f}")
    print(f"   Mediana: {np.median(proba):.3f}")
    
    # Analizar distribución
    print(f"\n📈 DISTRIBUCIÓN DE PROBABILIDADES:")
    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    range_labels = ["0.0-0.3", "0.3-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    
    cumulative = 0
    for i, (low, high) in enumerate(ranges):
        count = ((proba >= low) & (proba < high)).sum()
        pct = (count / len(proba)) * 100
        cumulative += pct
        print(f"   {range_labels[i]}: {count:3d} casos ({pct:5.1f}%) | Acumulado: {cumulative:5.1f}%")
    
    print(f"\n🎯 BUSCANDO THRESHOLD PARA 30-50% POSITIVOS...")
    print("-" * 60)
    
    # Rangos objetivo (de menos a más positivos)
    target_ranges = [
        (0.25, 0.35, "30-35% - Conservador"),
        (0.35, 0.45, "35-45% - Balanceado"), 
        (0.45, 0.55, "45-55% - Moderado")
    ]
    
    best_candidates = []
    
    for target_min, target_max, description in target_ranges:
        print(f"\n🎯 Objetivo: {description}")
        print("-" * 40)
        
        # Probar thresholds en el rango relevante
        if target_min < 0.35:
            thresholds = np.arange(0.7, 0.9, 0.02)  # Para pocos positivos
        elif target_min < 0.45:
            thresholds = np.arange(0.6, 0.8, 0.02)  # Para balance medio
        else:
            thresholds = np.arange(0.5, 0.7, 0.02)  # Para más positivos
        
        found = False
        for thr in thresholds:
            pred = (proba >= thr).astype(int)
            pos_rate = pred.mean()
            pos_count = pred.sum()
            
            if target_min <= pos_rate <= target_max:
                print(f"✅ thr={thr:.3f} -> {pos_count:3d} positivos ({pos_rate*100:5.1f}%)")
                
                # Guardar este candidato
                best_candidates.append({
                    'threshold': thr,
                    'positive_rate': pos_rate,
                    'positive_count': pos_count,
                    'description': description
                })
                found = True
        
        if not found:
            # Encontrar el más cercano
            closest_thr = None
            closest_diff = float('inf')
            
            for thr in thresholds:
                pred = (proba >= thr).astype(int)
                pos_rate = pred.mean()
                diff = abs(pos_rate - (target_min + target_max)/2)
                
                if diff < closest_diff:
                    closest_diff = diff
                    closest_thr = thr
                    closest_rate = pos_rate
                    closest_count = pred.sum()
            
            if closest_thr:
                print(f"⚠️  thr={closest_thr:.3f} -> {closest_count:3d} positivos ({closest_rate*100:5.1f}%) [MÁS CERCANO]")
                best_candidates.append({
                    'threshold': closest_thr,
                    'positive_rate': closest_rate, 
                    'positive_count': closest_count,
                    'description': description + " [CERCANO]"
                })
    
    # Mostrar mejores opciones
    if best_candidates:
        print(f"\n" + "=" * 70)
        print("🏆 MEJORES OPCIONES ENCONTRADAS:")
        print("=" * 70)
        
        best_candidates.sort(key=lambda x: abs(x['positive_rate'] - 0.4))  # Ordenar por cercanía al 40%
        
        for i, candidate in enumerate(best_candidates[:5], 1):
            print(f"{i}. {candidate['description']}")
            print(f"   Threshold: {candidate['threshold']:.3f}")
            print(f"   Resultado: {candidate['positive_count']} positivos ({candidate['positive_rate']*100:.1f}%)")
            
            # Crear y guardar submission para esta opción
            pred = (proba >= candidate['threshold']).astype(int)
            results = pd.DataFrame({"purchase_id": ids, "prediction": pred})
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outname = f"submission_optimal_{candidate['threshold']:.3f}_{ts}.csv"
            outpath = RESULTS_DIR / outname
            results.to_csv(outpath, index=False)
            print(f"   📁 Guardado: {outpath.name}")
            print()
    
    else:
        print("❌ No se encontraron thresholds adecuados")
        
        # Forzar creación con threshold de 0.6 como último recurso
        print("\n🔄 Creando submission con threshold=0.6 como último recurso...")
        thr = 0.6
        pred = (proba >= thr).astype(int)
        pos_rate = pred.mean()
        pos_count = pred.sum()
        
        print(f"✅ thr={thr:.3f} -> {pos_count} positivos ({pos_rate*100:.1f}%)")
        
        results = pd.DataFrame({"purchase_id": ids, "prediction": pred})
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outname = f"submission_emergency_{thr:.3f}_{ts}.csv"
        outpath = RESULTS_DIR / outname
        results.to_csv(outpath, index=False)
        print(f"📁 Guardado: {outpath.name}")

    print("=" * 70)
    print("✅ BÚSQUEDA COMPLETADA")
    print("=" * 70)

if __name__ == "__main__":
    find_optimal_threshold()