"""Detector trivial para tests: distancia al centro (z-score medio)."""
from __future__ import annotations

import numpy as np


class MockDetector:
    """Score = media del |z-score| por fila. No usa librerías de modelado."""

    nombre = "mock"

    def __init__(self) -> None:
        self._media: np.ndarray | None = None
        self._desv: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MockDetector":
        self._media = X.mean(axis=0)
        self._desv = X.std(axis=0) + 1e-9
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._media is None or self._desv is None:
            raise RuntimeError("El detector no está ajustado")
        return np.abs((X - self._media) / self._desv).mean(axis=1)
