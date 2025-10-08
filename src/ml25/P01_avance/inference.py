import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np

# Usa los dunders correctos
CURRENT_FILE = Path(__file__).resolve()
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR = CURRENT_FILE.parent / "trained_models"

def load_model(filename: str):
    filepath = Path(MODELS_DIR) / filename
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def load_test_data_with_labels():
    data_path = CURRENT_FILE.parent / "train_df_full.csv"
    if data_path.exists():
        data = pd.read_csv(data_path)
        X = data.drop(['customer_id_num', 'item_id_num', 'label'], axis=1, errors='ignore')
        y = data['label']
        print(f"Test data loaded: {X.shape}, Labels: {y.shape}")
        print(f"True label distribution: {y.value_counts().to_dict()}")
        return X, y
    else:
        print("train_df_full.csv not found")
        return None, None

def run_inference(model_name: str, X):
    model = load_model(model_name)
    preds = model.predict(X)
    # Asegura que el modelo soporte predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # fallback: decision_function -> escala a [0,1]
        scores = model.decision_function(X)
        probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return preds, probs

def evaluate_model(y_true, y_pred, y_proba, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Más simple y robusto:
    try:
        auc_score = roc_auc_score(y_true, y_proba)
    except ValueError:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)

    print(f"\nModel Evaluation - {model_name}:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc_score:.4f}")
    
    # Confusion matrix (y archivo)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    cm_path = RESULTS_DIR / f"cm_{model_name.replace('.pkl','')}.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # ROC curve (y archivo)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc='lower right')
    roc_path = RESULTS_DIR / f"roc_{model_name.replace('.pkl','')}.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_proba
    }

def get_all_models():
    if not MODELS_DIR.exists():
        return []
    model_files = list(MODELS_DIR.glob("*.pkl"))
    return [f.name for f in model_files]

def compare_models(metrics_dict):
    print("\n" + "="*50)
    print("COMPARACIÓN DE MODELOS")
    print("="*50)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        print(f"\n{metric.upper():>12}:", end="")
        for model_name, metrics in metrics_dict.items():
            print(f"  {model_name[:15]:15} {metrics[metric]:.4f}", end="")
    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n\nMEJOR MODELO: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")

if __name__ == "__main__":
    X, y_true = load_test_data_with_labels()
    if X is not None and y_true is not None:
        all_models = get_all_models()
        if all_models:
            print(f"\nFound {len(all_models)} models: {all_models}")
            metrics_results = {}
            for model_name in all_models:
                print(f"\n{'='*50}")
                print(f"EVALUANDO: {model_name}")
                print('='*50)
                y_pred, y_proba = run_inference(model_name, X)
                metrics = evaluate_model(y_true, y_pred, y_proba, model_name)
                metrics_results[model_name] = metrics
                filename = f"predictions_{model_name.replace('.pkl', '')}.csv"
                basepath = RESULTS_DIR / filename
                results_df = pd.DataFrame({
                    "ID": X.index,
                    "true_label": y_true,
                    "prediction": y_pred,
                    "probability": y_proba
                })
                results_df.to_csv(basepath, index=False)
                print(f"Saved predictions to {basepath}")
                print(f"Prediction stats - {model_name}:")
                print(f"  Positive predictions: {int(y_pred.sum())}/{len(y_pred)}")
                print(f"  Positive rate: {y_pred.mean():.4f}")
            compare_models(metrics_results)
        else:
            print("No trained models found")
    else:
        print("No test data available")
