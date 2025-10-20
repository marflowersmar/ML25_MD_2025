# model_xgboost.py — versión simple, sesgada a 0 y 100% compatible

from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
from xgboost import XGBClassifier

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent.parent / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        Xc = X.fillna(0); Xc = Xc.values if hasattr(X, "values") else Xc
        yb = np.asarray(y).astype(int)
        self.model.fit(Xc, yb)
        return self

    def predict_proba(self, X):
        Xc = X.fillna(0); Xc = Xc.values if hasattr(X, "values") else Xc
        return self.model.predict_proba(Xc)

    def save(self, prefix: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (MODELS_DIR / f"{prefix}_{ts}.pkl").resolve()
        joblib.dump(self, str(path))
        print(f"Model saved to {path}")
        return path

class XGBoostModel(BaseModel):
    """
    Preset conservador → aprende más 0 (menos 1s):
    - Árboles poco profundos, splits exigentes, regularización fuerte y prior bajo.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        p = {
            "n_estimators": 1200,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.70,
            "colsample_bytree": 0.70,
            "min_child_weight": 24.0,
            "gamma": 8.0,
            "reg_lambda": 12.0,
            "reg_alpha": 5.0,
            "max_delta_step": 3,
            "base_score": 0.20,          # prior bajo → favorece 0
            "scale_pos_weight": 0.40,    # peso fijo <= 1 → favorece 0
            "objective": "binary:logistic",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        p.update(kwargs)
        self.model = XGBClassifier(**p)
        self.kwargs = p

    def prefix_name(self):
        p = self.kwargs; s = lambda x: str(x).replace(".","p")
        return (f"xgb_n{p['n_estimators']}_md{p['max_depth']}_lr{s(p['learning_rate'])}"
                f"_ss{s(p['subsample'])}_cs{s(p['colsample_bytree'])}_mcw{s(p['min_child_weight'])}"
                f"_ga{s(p.get('gamma',0))}_rl{s(p['reg_lambda'])}_ra{s(p['reg_alpha'])}"
                f"_mds{s(p['max_delta_step'])}_bs{s(p['base_score'])}_spw{s(p['scale_pos_weight'])}")
