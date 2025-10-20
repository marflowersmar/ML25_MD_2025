# training_xgboost.py — simple y estable (sin callbacks/booster internos)
"""
- Modelo conservador (sesgo a 0) definido en model_xgboost.py
- Métricas @0.5 (reporte)
- Umbrales en VALIDACIÓN:
    * thr_youden
    * thr_f1
    * thr_final = threshold para dejar ~TARGET_POS_RATE de 1s en VALIDACIÓN
  (Se guarda y luego se usa tal cual en test; NO se toca la tasa del test)
"""

from pathlib import Path
import sys, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score,
    average_precision_score, precision_recall_curve, roc_curve
)

CURRENT_FILE = Path(__file__).resolve()
sys.path.append(str(CURRENT_FILE.parent.parent))

from utils import setup_logger
from data_processing import build_training_table, preprocess
from model_xgboost import XGBoostModel

# Objetivo de % de 1s EN VALIDACIÓN (no en test)
TARGET_POS_RATE = 0.25  # 25% → más 0s

def _youden_threshold(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr; i = int(np.nanargmax(j))
    return float(np.clip(thr[min(i, len(thr)-1)], 0.01, 0.99))

def _f1_threshold(y_true, y_proba):
    p, r, th = precision_recall_curve(y_true, y_proba)
    f1 = 2*p*r/(p+r+1e-9); i = int(np.nanargmax(f1))
    t = th[min(i, len(th)-1)] if len(th) else 0.5
    return float(np.clip(t, 0.01, 0.99))

def _threshold_by_target_rate(y_proba, target_rate):
    target_rate = float(max(0.01, min(0.99, target_rate)))
    rng = np.random.RandomState(42)
    jitter = (rng.rand(len(y_proba)) - 0.5) * 1e-6  # rompe empates
    thr = float(np.quantile(y_proba + jitter, 1 - target_rate))
    return float(np.clip(thr, 0.01, 0.99))

def train_xgboost(test_size=0.2, **model_params):
    rs = int(model_params.pop("random_state", 42))
    logger = setup_logger("xgboost_training")
    logger.info("XGBoost - Entrenamiento iniciado")

    # 1) Datos crudos de tu pipeline (no tocamos DP/NG)
    X_raw, y = build_training_table(n_per_positive=2, smart=True, random_state=rs)
    logger.info(f"Dataset crudo: X={X_raw.shape}, y+={y.mean():.3f}")

    # 2) Split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=rs
    )
    logger.info(f"Train: {len(X_tr_raw)} | Val: {len(X_va_raw)}")

    # 3) Preprocess
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)

    # 4) Modelo (ya sesgado a 0)
    model = XGBoostModel(random_state=rs, **model_params)
    logger.info("Entrenando...")
    model.fit(X_tr, y_tr)
    logger.info("Entrenamiento completado.")

    # 5) Métricas @0.5 (solo reporte)
    y_proba = model.predict_proba(X_va)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_va, y_pred)
    f1  = f1_score(y_va, y_pred)
    auc = roc_auc_score(y_va, y_proba)
    ap  = average_precision_score(y_va, y_proba)
    logger.info(f"Acc: {acc:.4f} | F1@0.5: {f1:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
    logger.info("\n" + classification_report(y_va, y_pred))

    # 6) Umbrales en VALIDACIÓN
    thr_y  = _youden_threshold(y_va, y_proba)
    thr_f1 = _f1_threshold(y_va, y_proba)
    thr_final = _threshold_by_target_rate(y_proba, TARGET_POS_RATE)

    rate_y   = float((y_proba >= thr_y).mean())
    rate_f1  = float((y_proba >= thr_f1).mean())
    rate_fin = float((y_proba >= thr_final).mean())
    logger.info(f"thr_youden={thr_y:.4f} -> %1s_val={rate_y*100:.2f}% | "
                f"thr_f1={thr_f1:.4f} -> %1s_val={rate_f1*100:.2f}% | "
                f"thr_final(TARGET {TARGET_POS_RATE*100:.0f}%)={thr_final:.4f} -> %1s_val={rate_fin*100:.2f}%")

    # 7) Guardar modelo + thresholds
    path = model.save(prefix=model.prefix_name())
    base = Path(str(path).replace(".pkl",""))
    (base.with_name(base.name + "_thr_final.txt")).write_text(str(thr_final))
    (base.with_name(base.name + "_thr_youden.txt")).write_text(str(thr_y))
    (base.with_name(base.name + "_thr_f1.txt")).write_text(str(thr_f1))
    logger.info(f"Modelo guardado: {path}")
    logger.info(f"Threshold FINAL (target {TARGET_POS_RATE*100:.0f}%) guardado junto al modelo.")

    return model, {
        "auc": auc, "f1@0.5": f1,
        "thr_final": thr_final, "thr_y": thr_y, "thr_f1": thr_f1,
        "final_pos_rate_val": rate_fin
    }

if __name__ == "__main__":
    print("="*80); print("XGBoost Training"); print("="*80)
    params = {}  # usa los defaults conservadores del modelo
    model, metrics = train_xgboost(test_size=0.2, **params)
    print("\n" + "="*80)
    print(f"AUC: {metrics['auc']:.4f} | F1@0.5: {metrics['f1@0.5']:.4f} | "
          f"thrFINAL: {metrics['thr_final']:.4f} (%1s_val={metrics['final_pos_rate_val']*100:.2f}%) | "
          f"thrY: {metrics['thr_y']:.4f} | thrF1: {metrics['thr_f1']:.4f}")
    print("="*80)
