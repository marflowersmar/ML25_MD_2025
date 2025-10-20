# training_rf.py
from pathlib import Path
import sys, numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score,
    average_precision_score, precision_recall_curve
)
import pandas as pd
import logging

# 🔒 SEMILLA GLOBAL PARA DETERMINISMO
GLOBAL_RANDOM_STATE = 42

def set_deterministic_environment():
    """Configura entorno completamente determinístico"""
    # 1. Semillas principales
    random.seed(GLOBAL_RANDOM_STATE)
    np.random.seed(GLOBAL_RANDOM_STATE)
    

# LLAMAR AL INICIO
set_deterministic_environment()

CURRENT_FILE = Path(__file__).resolve()
sys.path.append(str(CURRENT_FILE.parent.parent))

from utils import setup_logger
from data_processing import build_training_table, preprocess, read_csv
from model_rf import RandomForestModel

def setup_ultra_logger():
    """Logger para entrenamiento ultra-seguro"""
    logger = logging.getLogger("ultra_safe_training")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def build_ultra_safe_training_data():
    """
    Construye datos de entrenamiento ULTRA-SEGUROS y DETERMINÍSTICOS
    """
    logger = setup_ultra_logger()
    logger.info("CONSTRUYENDO DATASET ULTRA-SEGURO Y DETERMINÍSTICO")
    
    # 🔒 SEMILLA ESPECÍFICA PARA ESTA FUNCIÓN
    random_state_func = GLOBAL_RANDOM_STATE
    
    # 1) Leer datos originales
    train_df = read_csv("customer_purchases_train")
    logger.info(f"Datos originales: {train_df.shape}")
    
    # 2) ELIMINAR SOLO COLUMNAS PELIGROSAS
    dangerous_columns = [
        'purchase_timestamp', 'customer_signup_date', 'item_release_date',
        'customer_item_views', 'purchase_item_rating', 'purchase_id'
    ]
    
    safe_columns = [
        'item_price', 'item_category', 'item_title',
        'customer_gender', 'customer_date_of_birth', 'customer_id'
    ]
    
    # Filtrar solo columnas seguras que existan
    safe_columns = [col for col in safe_columns if col in train_df.columns]
    logger.info(f"Columnas ultra-seguras: {safe_columns}")
    
    # 3) Customer features básicos (determinísticos)
    logger.info("Calculando features básicos de cliente...")
    
    # 🔒 Groupby determinístico (usando sort=False para consistencia)
    customer_basic_features = train_df.groupby('customer_id', sort=False).agg({
        'item_price': ['count', 'mean', 'std', 'min', 'max'],
        'item_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
    }).reset_index()
    
    customer_basic_features.columns = [
        'customer_id', 
        'customer_purchase_count', 'customer_avg_spent', 
        'customer_std_spent', 'customer_min_spent', 'customer_max_spent',
        'customer_fav_category'
    ]
    
    # 4) Positivos con features básicos
    pos = train_df[safe_columns].copy()
    pos = pos.merge(customer_basic_features, on="customer_id", how="left")
    pos["label"] = 1

    # 5) Negativos - CON SEMILLA EXPLÍCITA
    from negative_generation import gen_random_negatives
    neg = gen_random_negatives(
        train_df, 
        n_per_positive=15, 
        smart=False, 
        random_state=random_state_func  # 🔒 SEMILLA EXPLÍCITA
    )
    
    # Enriquecer negativos con información mínima
    item_basic_cols = ['item_id', 'item_price', 'item_category', 'item_title']
    item_tbl = train_df[item_basic_cols].drop_duplicates("item_id")
    
    neg = neg.merge(customer_basic_features, on="customer_id", how="left")
    neg = neg.merge(item_tbl, on="item_id", how="left")

    # Fallbacks seguros
    neg["item_category"] = neg["item_category"].fillna("unknown_category").astype(str)
    neg["item_title"] = neg["item_title"].fillna("").astype(str)
    neg["item_price"] = pd.to_numeric(neg["item_price"], errors="coerce").fillna(train_df["item_price"].median())

    # 6) Concat + shuffle DETERMINÍSTICO
    full = pd.concat([pos, neg], ignore_index=True)
    full = full.sample(frac=1, random_state=random_state_func)  # 🔒 SEMILLA EXPLÍCITA
    
    logger.info(f"Dataset ultra-seguro: {full.shape}")
    logger.info(f"Balance: {full['label'].mean():.3f} (1's)")

    # X_raw y y - eliminar solo IDs peligrosos
    y = full["label"].copy()
    X_raw = full.drop(columns=["label", "customer_id", "item_id", "purchase_id"], errors="ignore")
    
    # Verificar columnas finales
    remaining_cols = list(X_raw.columns)
    logger.info(f"Columnas finales: {len(remaining_cols)}")
    
    return X_raw, y

def train_ultra_safe_random_forest(test_size=0.3, **model_params):
    """
    Entrenamiento ULTRA-SEGURO Y DETERMINÍSTICO
    """
    # 🔒 USAR SEMILLA GLOBAL SI NO SE ESPECIFICA
    random_state = int(model_params.pop("random_state", GLOBAL_RANDOM_STATE))
    logger = setup_ultra_logger()
    logger.info(f"RANDOM FOREST - ENTRENAMIENTO DETERMINÍSTICO (random_state={random_state})")

    # 1) Datos ultra-seguros
    X_raw, y = build_ultra_safe_training_data()
    logger.info(f"Dataset ultra-seguro: {X_raw.shape}, balance: {y.mean():.3f}")

    # 2) Split DETERMINÍSTICO
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state  # 🔒 SEMILLA EXPLÍCITA
    )
    logger.info(f"Train: {X_tr_raw.shape[0]} | Val: {X_va_raw.shape[0]}")
    logger.info(f"Balance train: {y_tr.mean():.3f} | Balance val: {y_va.mean():.3f}")

    # 3) Preprocess (asegurar que sea determinístico)
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)

    # 4) MODELO DETERMINÍSTICO
    # 🔒 FORZAR random_state en todos los parámetros posibles
    model_params['random_state'] = random_state
    
    logger.info(f"Parámetros del modelo (DETERMINÍSTICOS): {model_params}")
    model = RandomForestModel(**model_params)
    
    logger.info("Entrenando modelo...")
    model.fit(X_tr, y_tr)
    logger.info("Entrenamiento completado.")

    # 5) EVALUACIÓN DETERMINÍSTICA
    y_proba = model.predict_proba(X_va)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Métricas básicas (siempre las mismas con mismos datos)
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred)
    roc = roc_auc_score(y_va, y_proba)
    
    logger.info(f"METRICAS VALIDATION (thr=0.5):")
    logger.info(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {roc:.4f}")
    logger.info(f"   % Positivos: {y_pred.mean():.3f}")
    
    # 6) THRESHOLDS DETERMINÍSTICOS
    precision, recall, ths = precision_recall_curve(y_va, y_proba)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)

    if len(ths) > 0:
        best_idx = int(np.nanargmax(f1s[:-1]))
        thr_f1 = float(ths[best_idx])
    else:
        thr_f1 = 0.5

    # Thresholds ultra-conservadores con clip [0.01, 0.99]
    thr_safe = float(np.clip(max(0.7, thr_f1 + 0.25), 0.01, 0.99))
    thr_balanced = float(np.clip(max(0.6, thr_f1 + 0.15), 0.01, 0.99))

    logger.info(f"THRESHOLDS CALCULADOS:")
    logger.info(f"   F1: {thr_f1:.4f}")
    logger.info(f"   BALANCED: {thr_balanced:.4f}")
    logger.info(f"   SAFE: {thr_safe:.4f}")

    # Verificar tasas (siempre las mismas)
    logger.info(f"Tasas de positivos en validation:")
    for thr_name, thr_val in [("F1", thr_f1), ("BALANCED", thr_balanced), ("SAFE", thr_safe)]:
        pos_rate = (y_proba >= thr_val).mean()
        logger.info(f"   {thr_name}({thr_val:.3f}): {pos_rate:.3f}")

    # 7) Guardar modelo con información de determinismo
    if y_pred.mean() <= 0.7:
        path = model.save(prefix=model.prefix_name() + f"_DETERMINISTIC_RS{random_state}")
        base = Path(str(path).replace(".pkl", ""))

        # Guardar thresholds
        (base.with_name(base.name + "_thr_f1.txt")).write_text(str(thr_f1))
        (base.with_name(base.name + "_thr_balanced.txt")).write_text(str(thr_balanced))
        (base.with_name(base.name + "_thr_safe.txt")).write_text(str(thr_safe))
        
        # Guardar info de determinismo
        determinism_info = f"random_state={random_state}\nglobal_seed={GLOBAL_RANDOM_STATE}"
        (base.with_name(base.name + "_determinism.txt")).write_text(determinism_info)

        logger.info(f"Modelo determinístico guardado: {path}")

        return model, {
            "accuracy": acc, "f1": f1, "roc_auc": roc,
            "thr_f1": thr_f1, "thr_balanced": thr_balanced, "thr_safe": thr_safe,
            "val_positive_rate": y_pred.mean(),
            "random_state": random_state
        }
    else:
        return None, None

if __name__ == "__main__":
    print("=" * 80)
    print("RANDOM FOREST TRAINING - DETERMINÍSTICO Y ULTRA-SEGURO")
    print(f"SEMILLA GLOBAL: {GLOBAL_RANDOM_STATE}")
    print("=" * 80)

    # PARÁMETROS DETERMINÍSTICOS
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 50,
        "min_samples_leaf": 25,
        "max_features": 0.2,
        "bootstrap": True,
        "class_weight": "balanced",
        "random_state": GLOBAL_RANDOM_STATE,  # 🔒 SEMILLA EXPLÍCITA
        "n_jobs": -1,
    }

    try:
        model, metrics = train_ultra_safe_random_forest(test_size=0.3, **params)

        if model is not None:
            print("\n" + "=" * 80)
            print("✅ ENTRENAMIENTO DETERMINÍSTICO COMPLETADO")
            print(f"SEMILLA: {metrics['random_state']}")
            print(f"AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
            print(f"Thresholds -> BALANCED: {metrics['thr_balanced']:.4f} | SAFE: {metrics['thr_safe']:.4f}")
            print(f"Tasa positivos en val: {metrics['val_positive_rate']:.3f}")
            
            if metrics['val_positive_rate'] <= 0.4:
                print("🎉 PERFECTO: Modelo bien calibrado y determinístico")
            else:
                print("💡 BUENO: Modelo funcionando correctamente y determinístico")
                
        else:
            print("\n❌ ENTRENAMIENTO FALLADO")
            
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR en entrenamiento: {e}")
        import traceback
        traceback.print_exc()