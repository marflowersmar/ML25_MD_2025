# training_rf.py
import os
from pathlib import Path
from datetime import datetime
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from model_rf import PurchaseModel
from data_processing import build_training_table, preprocess

# -----------------------------------------------------------
# Configuración ORIGINAL MEJORADA
# -----------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
NEG_PER_POS = 3  # Balance razonable: 3 negativos por positivo
NEG_SMART = True

GRID = {
    "n_estimators": [800, 1200],  # Mantener poder predictivo
    "max_depth": [20, 25],  # Permitir más profundidad
    "min_samples_leaf": [1, 2],  # Menos regularización
    "max_features": ["sqrt"],
    "class_weight": ["balanced", None],  # Probar con y sin balance
}

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------
def param_product(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))

def make_model_name(params: dict, calibrated: bool) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cw_str = "balanced" if params.get('class_weight') == "balanced" else "none"
    base = f"rf_opt_n{params['n_estimators']}_md{params['max_depth']}_msl{params['min_samples_leaf']}_{cw_str}_{ts}"
    return f"{base}_CAL.pkl" if calibrated else f"{base}.pkl"

def compute_metrics(model: PurchaseModel, X_tr, y_tr, X_va, y_va, threshold=0.5):
    p_tr = model.predict_proba(X_tr)
    p_va = model.predict_proba(X_va)
    yhat_va = (p_va >= threshold).astype(int)

    return {
        "roc_tr": float(roc_auc_score(y_tr, p_tr)),
        "roc_val": float(roc_auc_score(y_va, p_va)),
        "avgprec_val": float(average_precision_score(y_va, p_va)),
        "precision_val": float(precision_score(y_va, yhat_va, zero_division=0)),
        "recall_val": float(recall_score(y_va, yhat_va, zero_division=0)),
        "f1_val": float(f1_score(y_va, yhat_va, zero_division=0)),
        "acc_val": float(accuracy_score(y_va, yhat_va)),
    }

# -----------------------------------------------------------
# Entrenamiento principal OPTIMIZADO
# -----------------------------------------------------------
def run_training():
    print("🚀 INICIANDO ENTRENAMIENTO OPTIMIZADO...")
    print("Construyendo tabla de entrenamiento...")
    X_raw, y = build_training_table(n_per_positive=NEG_PER_POS, smart=NEG_SMART, random_state=RANDOM_STATE)
    print(f"📊 Dataset - Filas: {len(X_raw)}  Pos: {int(y.sum())}  Neg: {int((1 - y).sum())}")
    print(f"🔢 Ratio: {int(y.sum())}:{int((1-y).sum())} ({y.mean():.1%} positivos)")

    # Holdout estratificado
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("Preprocesando...")
    X_tr = preprocess(X_tr_raw, training=True)
    X_val = preprocess(X_val_raw, training=False)

    results = []
    best_f1 = 0
    best_model = None
    best_params = None
    best_threshold = 0.5

    print("🔍 Buscando mejores hiperparámetros...")
    for i, params in enumerate(param_product(GRID)):
        print(f"  Probando {i+1}/8: {params}")
        
        model = PurchaseModel(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=2,
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        
        # Calcular métricas con threshold 0.5
        m = compute_metrics(model, X_tr, y_tr, X_val, y_val, threshold=0.5)
        results.append({"calibrated": False, **params, **m})
        
        # Encontrar mejor threshold para F1
        p_val = model.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        current_best_f1 = f1_scores[best_idx]
        current_threshold = thresholds[best_idx]
        
        if current_best_f1 > best_f1:
            best_f1 = current_best_f1
            best_model = model
            best_params = params
            best_threshold = current_threshold
            print(f"    ✅ Nuevo mejor: F1={current_best_f1:.4f}, th={current_threshold:.4f}")

    # Calibración del mejor modelo
    print(f"\n🎯 Calibrando mejor modelo (F1={best_f1:.4f})...")
    cal = CalibratedClassifierCV(best_model.clf, method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    best_model.clf = cal

    # Recalcular métricas después de calibración
    p_val_cal = best_model.predict_proba(X_val)
    precisions_cal, recalls_cal, thresholds_cal = precision_recall_curve(y_val, p_val_cal)
    f1_scores_cal = 2 * (precisions_cal[:-1] * recalls_cal[:-1]) / (precisions_cal[:-1] + recalls_cal[:-1] + 1e-8)
    best_idx_cal = np.argmax(f1_scores_cal)
    best_threshold_cal = thresholds_cal[best_idx_cal]
    best_f1_cal = f1_scores_cal[best_idx_cal]

    m_cal = compute_metrics(best_model, X_tr, y_tr, X_val, y_val, threshold=best_threshold_cal)
    results.append({"calibrated": True, **best_params, **m_cal})

    # Resultados finales
    df = pd.DataFrame(results)
    df = df.sort_values(by=["calibrated", "f1_val"], ascending=[True, False])

    print("\n" + "="*60)
    print("📊 MEJORES RESULTADOS")
    print("="*60)
    print(df[["calibrated", "n_estimators", "max_depth", "min_samples_leaf", 
              "class_weight", "roc_val", "f1_val", "precision_val", "recall_val"]].round(4))

    # Guardar todo
    models_dir = Path(__file__).resolve().parent / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar threshold óptimo (usar el de la calibración)
    optimal_threshold = best_threshold_cal
    opt_path = models_dir / "optimal_threshold.txt"
    with open(opt_path, "w", encoding="utf-8") as f:
        f.write(f"{optimal_threshold:.6f}\n")
    
    print(f"\n🎯 Umbral óptimo CALIBRADO: {optimal_threshold:.4f}")
    print(f"📈 F1 score calibrado: {best_f1_cal:.4f}")

    # Guardar modelo
    model_path = models_dir / make_model_name(best_params, calibrated=True)
    best_model.save(str(model_path))
    print(f"💾 Modelo guardado: {model_path}")

    # Distribución final en validation
    y_val_final = (p_val_cal >= optimal_threshold).astype(int)
    unique, counts = np.unique(y_val_final, return_counts=True)
    print(f"\n📊 DISTRIBUCIÓN EN VALIDACIÓN:")
    for val, count in zip(unique, counts):
        print(f"   Clase {val}: {count} ({count/len(y_val_final):.1%})")

    return {
        "best_params": best_params, 
        "best_threshold": optimal_threshold,
        "best_f1": best_f1_cal,
        "model_path": str(model_path)
    }

if __name__ == "__main__":
    print("🎬 INICIANDO ENTRENAMIENTO OPTIMIZADO")
    results = run_training()
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"🏆 Mejor F1: {results['best_f1']:.4f}")
    print(f"🎯 Threshold: {results['best_threshold']:.4f}")
    print(f"📁 Modelo: {results['model_path']}")