"""Isolation Forest ajustado solo con filas normales."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Score = -score_samples (mayor = más anómalo)."""

    nombre = "iforest"

    def __init__(self, n_arboles: int = 200, seed: int = 42) -> None:
        self._m = IsolationForest(n_estimators=n_arboles, random_state=seed, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestDetector":
        self._m.fit(X if y is None else X[y == 0])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._m.score_samples(X)
