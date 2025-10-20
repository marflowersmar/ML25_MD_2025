# training_rf_emergency.py
from pathlib import Path
import sys, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score,
    average_precision_score, precision_recall_curve
)
import pandas as pd
import logging

CURRENT_FILE = Path(__file__).resolve()
sys.path.append(str(CURRENT_FILE.parent.parent))

from utils import setup_logger
from data_processing import build_training_table, preprocess, read_csv
from model_rf import RandomForestModel


def setup_safe_logger():
    """Logger sin emojis para evitar problemas de encoding"""
    logger = logging.getLogger("safe_training")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def inspect_data_leakage():
    """Inspecciona posibles fugas de datos"""
    logger = setup_safe_logger()
    logger.info("INSPECCION DE FUGAS DE DATOS")
    
    # Leer datos originales
    train_df = read_csv("customer_purchases_train")
    
    # Verificar columnas problemáticas
    problematic_cols = []
    for col in train_df.columns:
        if 'test' in col.lower() or 'future' in col.lower() or 'target' in col.lower():
            problematic_cols.append(col)
    
    if problematic_cols:
        logger.warning(f"COLUMNAS POTENCIALMENTE PROBLEMATICAS: {problematic_cols}")
    
    # Verificar si hay información del futuro
    date_cols = [col for col in train_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    logger.info(f"Columnas de fecha: {date_cols}")
    
    return problematic_cols


def build_safe_training_table():
    """
    Construye tabla de entrenamiento ELIMINANDO posibles fugas
    """
    logger = setup_safe_logger()
    
    # 1) Leer datos originales
    train_df = read_csv("customer_purchases_train")
    
    # 2) ELIMINAR COLUMNAS PELIGROSAS que podrían causar fuga
    dangerous_columns = [
        # Columnas que podrían contener información del futuro
        'purchase_timestamp', 'customer_signup_date', 'item_release_date',
        # Columnas que podrían ser targets leak
        'customer_item_views', 'purchase_item_rating', 
        # IDs que no deben usarse como features
        'purchase_id'
    ]
    
    # Mantener solo columnas seguras (PERO MANTENER customer_id para el merge)
    safe_columns = [col for col in train_df.columns if col not in dangerous_columns]
    logger.info(f"Usando {len(safe_columns)} columnas seguras de {len(train_df.columns)} totales")
    logger.info(f"Columnas seguras: {safe_columns}")
    
    # 3) Customer features (sin información temporal peligrosa)
    from data_processing import extract_customer_features
    customer_feat = extract_customer_features(train_df)
    
    # 4) Positivos del histórico (SOLO con columnas seguras, PERO mantener customer_id)
    pos = train_df[safe_columns].copy()
    
    # VERIFICAR que customer_id existe antes del merge
    if 'customer_id' not in pos.columns:
        logger.error("ERROR: customer_id no encontrado en columnas seguras")
        logger.error(f"Columnas disponibles: {list(pos.columns)}")
        raise KeyError("customer_id no disponible para merge")
        
    pos = pos.merge(customer_feat, on="customer_id", how="left")
    pos["label"] = 1

    # 5) Negativos - MUCHOS MAS
    from negative_generation import gen_random_negatives
    neg = gen_random_negatives(train_df, n_per_positive=15, smart=True, random_state=42)
    
    # Enriquecer negativos con información segura
    item_safe_cols = ['item_id', 'item_price', 'item_category', 'item_title']
    item_tbl = train_df[item_safe_cols].drop_duplicates("item_id")
    
    neg = neg.merge(customer_feat, on="customer_id", how="left")
    neg = neg.merge(item_tbl, on="item_id", how="left")

    # Fallbacks seguros
    neg["item_category"] = neg["item_category"].fillna("unknown_category").astype(str)
    neg["item_title"] = neg["item_title"].fillna("").astype(str)
    neg["item_price"] = pd.to_numeric(neg["item_price"], errors="coerce").fillna(train_df["item_price"].median())

    # 6) Concat + shuffle
    full = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)
    
    logger.info(f"Dataset final: {full.shape}")
    logger.info(f"Balance: {full['label'].mean():.3f} (1's)")

    # X_raw y y sin IDs (PERO solo despues de todos los merges)
    y = full["label"].copy()
    X_raw = full.drop(columns=["label", "customer_id", "item_id", "purchase_id"], errors="ignore")
    
    return X_raw, y


def train_safe_random_forest():
    """
    Entrenamiento SEGURO sin fugas de datos
    """
    logger = setup_safe_logger()
    logger.info("ENTRENAMIENTO SEGURO INICIADO")
    
    # 1) Inspeccionar fugas
    inspect_data_leakage()
    
    # 2) Dataset seguro
    X_raw, y = build_safe_training_table()
    logger.info(f"Dataset seguro: {X_raw.shape}, balance: {y.mean():.3f}")

    # 3) Split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42
    )
    logger.info(f"Train: {X_tr_raw.shape[0]} | Val: {X_va_raw.shape[0]}")
    logger.info(f"Balance train: {y_tr.mean():.3f} | Balance val: {y_va.mean():.3f}")

    # 4) Preprocess
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)

    # 5) MODELO MUY SIMPLE Y RESTRICTIVO
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "min_samples_split": 100,
        "min_samples_leaf": 50,
        "max_features": 0.1,
        "bootstrap": True,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    
    logger.info(f"Parametros ULTRA-RESTRICTIVOS: {params}")
    model = RandomForestModel(**params)
    
    logger.info("Entrenando modelo...")
    model.fit(X_tr, y_tr)
    logger.info("Entrenamiento completado")

    # 6) Evaluación
    y_proba = model.predict_proba(X_va)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Métricas básicas
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred)
    roc = roc_auc_score(y_va, y_proba)
    
    logger.info(f"Metricas Validation (thr=0.5):")
    logger.info(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {roc:.4f}")
    logger.info(f"   % Positivos: {y_pred.mean():.3f}")
    logger.info("\n" + classification_report(y_va, y_pred))

    # 7) THRESHOLDS MUY ALTOS
    precision, recall, ths = precision_recall_curve(y_va, y_proba)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    
    if len(ths) > 0:
        best_idx = int(np.nanargmax(f1s[:-1]))
        thr_f1 = float(ths[best_idx])
    else:
        thr_f1 = 0.5
    
    # Thresholds progresivamente más altos
    thr_conservative = max(0.7, thr_f1 + 0.2)
    thr_ultra = max(0.8, thr_f1 + 0.3)
    thr_extreme = max(0.85, thr_f1 + 0.4)
    
    logger.info(f"THRESHOLDS CALCULADOS:")
    logger.info(f"   F1: {thr_f1:.4f}")
    logger.info(f"   Conservative: {thr_conservative:.4f}")
    logger.info(f"   Ultra: {thr_ultra:.4f}")
    logger.info(f"   EXTREME: {thr_extreme:.4f}")

    # Verificar tasas con cada threshold
    logger.info(f"TASAS DE POSITIVOS EN VALIDATION:")
    for thr_name, thr_val in [("F1", thr_f1), ("CONS", thr_conservative), 
                             ("ULTRA", thr_ultra), ("EXTREME", thr_extreme)]:
        pos_rate = (y_proba >= thr_val).mean()
        logger.info(f"   {thr_name}({thr_val:.3f}): {pos_rate:.3f}")

    # 8) Guardar modelo
    path = model.save(prefix="rf_SAFE_emergency")
    base = Path(str(path).replace(".pkl", ""))
    
    # Guardar todos los thresholds
    (base.with_name(base.name + "_thr_f1.txt")).write_text(str(thr_f1))
    (base.with_name(base.name + "_thr_conservative.txt")).write_text(str(thr_conservative))
    (base.with_name(base.name + "_thr_ultra.txt")).write_text(str(thr_ultra))
    (base.with_name(base.name + "_thr_extreme.txt")).write_text(str(thr_extreme))
    
    logger.info(f"Modelo guardado: {path}")

    return model, {
        "accuracy": acc, "f1": f1, "roc_auc": roc,
        "thr_f1": thr_f1, "thr_conservative": thr_conservative, 
        "thr_ultra": thr_ultra, "thr_extreme": thr_extreme,
        "val_positive_rate": y_pred.mean(),
        "val_proba_mean": y_proba.mean()
    }


if __name__ == "__main__":
    print("=" * 80)
    print("RANDOM FOREST - ENTRENAMIENTO DE EMERGENCIA")
    print("=" * 80)
    
    try:
        model, metrics = train_safe_random_forest()
        
        print("\n" + "=" * 80)
        print("ENTRENAMIENTO DE EMERGENCIA COMPLETADO")
        print(f"Metricas Validation:")
        print(f"   AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"   % Positivos (thr=0.5): {metrics['val_positive_rate']:.3f}")
        print(f"Thresholds disponibles:")
        print(f"   F1: {metrics['thr_f1']:.4f}")
        print(f"   CONS: {metrics['thr_conservative']:.4f}")
        print(f"   ULTRA: {metrics['thr_ultra']:.4f}")
        print(f"   EXTREME: {metrics['thr_extreme']:.4f}")
        print("USAR 'extreme' EN INFERENCE PARA MEJOR BALANCE")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR en entrenamiento: {e}")
        import traceback
        traceback.print_exc()