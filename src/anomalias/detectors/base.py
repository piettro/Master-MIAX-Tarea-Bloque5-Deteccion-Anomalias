"""Contrato común de los detectores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    """Detector de anomalías: mayor score = más anómalo."""

    nombre: str

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Detector":
        """Ajusta el detector (y es opcional)."""
        ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """Devuelve un score por fila (longitud = len(X))."""
        ...


@dataclass(frozen=True)
class EspecDetector:
    """Receta de un detector para el orquestador (fabrica recibe la semilla)."""

    nombre: str
    fabrica: Callable[[int], Detector]
    opcional: bool = True
    peso: float = 1.0
