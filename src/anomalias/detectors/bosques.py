"""Bosques aleatorios supervisados (RandomForest y ExtraTrees) con clases balanceadas."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


class _Bosque:
    """Base común: score = probabilidad de fraude."""

    nombre = "bosque"
    _clase: type = RandomForestClassifier

    def __init__(self, seed: int = 42, n_arboles: int = 300) -> None:
        self._m = self._clase(
            n_estimators=n_arboles,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_Bosque":
        if y is None:
            raise ValueError(f"{self.nombre} necesita etiquetas")
        self._m.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict_proba(X)[:, 1]


class RandomForestDetector(_Bosque):
    nombre = "rf"
    _clase = RandomForestClassifier


class ExtraTreesDetector(_Bosque):
    nombre = "et"
    _clase = ExtraTreesClassifier
