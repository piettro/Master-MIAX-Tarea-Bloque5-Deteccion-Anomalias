"""Mezcla gaussiana ajustada solo con filas normales (baseline del notebook de partida)."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMDetector:
    """Score = -log-verosimilitud (mayor = más anómalo)."""

    def __init__(self, n_componentes: int = 3, seed: int = 42) -> None:
        self.nombre = f"gmm{n_componentes}"
        self._m = GaussianMixture(n_components=n_componentes, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GMMDetector":
        self._m.fit(X if y is None else X[y == 0])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._m.score_samples(X)
