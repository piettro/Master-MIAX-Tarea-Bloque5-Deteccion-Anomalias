"""Gradient boosting supervisado (baseline con etiquetas, P-02)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class GBMSupervisado:
    """Score = probabilidad de fraude (mayor = más anómalo).

    La regularización (l2, hojas mínimas y pocas hojas) es clave: con los valores por
    defecto de sklearn el AUPR colapsa y es inestable (ver docs/03_DECISIONES.md).
    """

    nombre = "hgb"

    def __init__(self, seed: int = 42, max_iter: int = 200) -> None:
        self._m = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=max_iter,
            max_leaf_nodes=8,
            min_samples_leaf=50,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GBMSupervisado":
        if y is None:
            raise ValueError("GBMSupervisado necesita etiquetas")
        self._m.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict_proba(X)[:, 1]
