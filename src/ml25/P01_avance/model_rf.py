import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


class PurchaseModel:
    """
    Wrapper sencillo para un clasificador sklearn.
    - self.clf puede ser RandomForestClassifier o CalibratedClassifierCV
    - Métodos robustos a calibración (predict_proba, save, __repr__)
    """

    def __init__(
        self,
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    ):
        # Normalizar class_weight si viene como dict con llaves string
        cw = class_weight
        if isinstance(cw, dict):
            try:
                cw = {int(k): float(v) for k, v in cw.items()}
            except Exception:
                # Si ya son enteros o no se pueden castear, lo dejamos tal cual
                pass

        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=cw,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.clf = RandomForestClassifier(**self.params)

    # -----------------------------------------------------
    # sklearn-like API
    # -----------------------------------------------------
    def fit(self, X, y):
        """Entrena el clasificador actual (RF si no lo has cambiado por un calibrado)."""
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Devuelve probas de clase positiva (columna 1 si binario).
        Funciona para RF y para CalibratedClassifierCV.
        """
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X)
            # Binario → columna de la clase 1
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        raise AttributeError(f"El clasificador {type(self.clf).__name__} no expone predict_proba.")

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -----------------------------------------------------
    # Persistencia
    # -----------------------------------------------------
    def save(self, filepath: str):
        """
        Guarda SOLO el estimador subyacente (self.clf).
        No guardamos el wrapper para evitar problemas de versiones.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, filepath)
        print(f"{self._safe_repr()} || Model saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """
        Carga el estimador y lo envuelve en PurchaseModel.
        Nota: al cargar no reconstruimos params originales del RF; no son necesarios para inferencia.
        """
        model = PurchaseModel()
        model.clf = joblib.load(filepath)
        return model

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _safe_repr(self) -> str:
        """
        Representación robusta tanto para RF como para CalibratedClassifierCV.
        - Si está calibrado, intentamos leer los params del base_estimator si existe.
        - Si no, devolvemos el nombre de clase y, si aplica, n_estimators.
        """
        name = type(self.clf).__name__

        # Caso calibrado
        if isinstance(self.clf, CalibratedClassifierCV):
            base = getattr(self.clf, "base_estimator", None)
            if base is None and hasattr(self.clf, "estimator"):
                base = getattr(self.clf, "estimator", None)

            if base is not None and hasattr(base, "get_params"):
                cfg = base.get_params()
                ne  = cfg.get("n_estimators", "?")
                md  = cfg.get("max_depth", "?")
                msl = cfg.get("min_samples_leaf", "?")
                mf  = cfg.get("max_features", "?")
                cw  = cfg.get("class_weight", "?")
                return f"CalibratedClassifierCV(base=RandomForest(n={ne}, depth={md}, leaf={msl}, max_feat={mf}, cw={cw}))"
            else:
                return f"CalibratedClassifierCV(base={type(base).__name__ if base is not None else 'Unknown'})"

        # Caso RF normal
        if isinstance(self.clf, RandomForestClassifier):
            cfg = self.clf.get_params()
            ne  = cfg.get("n_estimators", "?")
            md  = cfg.get("max_depth", "?")
            msl = cfg.get("min_samples_leaf", "?")
            mf  = cfg.get("max_features", "?")
            cw  = cfg.get("class_weight", "?")
            return f"RandomForest(n={ne}, depth={md}, leaf={msl}, max_feat={mf}, cw={cw})"

        # Cualquier otro estimador
        return name

    def __repr__(self):
        return self._safe_repr()
