# training_rf.py
# Entrenamiento Random Forest sin calibración para Customer Purchases

# ML
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

# Custom locales
from utils import setup_logger
from data_processing import build_training_table, preprocess
from model_rf import PurchaseModel


CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
TRAINED_DIR = BASE_DIR / "trained_models"
DATASETS_DIR = BASE_DIR.parent / "datasets" / "customer_purchases"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)


def pick_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    ths = np.linspace(0.05, 0.95, 37)
    f1s = []
    for th in ths:
        y_pred = (y_proba >= th).astype(int)
        f1s.append(f1_score(y_true, y_pred))
    j = int(np.argmax(f1s))
    return float(ths[j])


def eval_metrics(y_true: np.ndarray, y_proba: np.ndarray, th: float) -> dict:
    y_pred = (y_proba >= th).astype(int)
    return dict(
        f1=f1_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        roc=roc_auc_score(y_true, y_proba),
        avgprec=average_precision_score(y_true, y_proba),
        acc=float((y_pred == y_true).mean()),
    )


def run_training(random_state: int = 42, n_per_positive: int = 1):
    logger = setup_logger("training_rf")
    logger.info("Iniciando entrenamiento RF sin calibración")

    # 1. Construir tabla de entrenamiento cruda
    X_raw, y = build_training_table(n_per_positive=n_per_positive, smart=True)
    y = np.asarray(y).astype(int)
    logger.info(f"Dataset  filas={len(y)}  pos={(y==1).sum()}  neg={(y==0).sum()}")

    # 2. Split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # 3. Preprocesar y guardar preprocessor.pkl desde data_processing
    logger.info("Preprocesando...")
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)
    logger.info(f"Dimensiones  X_tr={X_tr.shape}  X_va={X_va.shape}")

    # 4. Entrenar modelo RF sin calibración
    #    Hiperparámetros sólidos para tu caso según runs previos
    model = PurchaseModel(
        n_estimators=800,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info(f"Entrenando {repr(model)}")
    model.fit(X_tr, y_tr)

    # 5. Probabilidades en validación y selección de threshold por F1
    proba_va = model.predict_proba(X_va)
    # Diagnóstico de probas
    q = np.percentile(proba_va, [0, 25, 50, 75, 100])
    print(
        f"Probas validación  min={q[0]:.4f} p25={q[1]:.4f} mediana={q[2]:.4f} "
        f"p75={q[3]:.4f} max={q[4]:.4f}"
    )

    th_opt = pick_best_threshold(y_va, proba_va)
    metrics = eval_metrics(y_va, proba_va, th_opt)
    pct_ge = 100.0 * (proba_va >= th_opt).mean()
    print(f"Threshold óptimo={th_opt:.4f}  porcentaje >= th={pct_ge:.1f}%")
    print(
        f"Métricas val  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  ROC={metrics['roc']:.4f}  AP={metrics['avgprec']:.4f}  "
        f"ACC={metrics['acc']:.4f}"
    )
    print("\nClassification report en validación:")
    print(classification_report(y_va, (proba_va >= th_opt).astype(int), digits=4))

    # 6. Guardar modelo y threshold
    cfg = model.get_config()
    prefix = f"rf_n{cfg['n_estimators']}_md{cfg['max_depth']}_msl{cfg['min_samples_leaf']}_mfsqrt_cw{cfg['class_weight']}"
    model_path = model.save(prefix=prefix)
    (TRAINED_DIR / "optimal_threshold.txt").write_text(f"{th_opt:.6f}")
    print(f"💾 Threshold guardado en: {TRAINED_DIR / 'optimal_threshold.txt'}")

    print("✅ ENTRENAMIENTO COMPLETADO")
    return str(model_path), th_opt, metrics


if __name__ == "__main__":
    run_training()
