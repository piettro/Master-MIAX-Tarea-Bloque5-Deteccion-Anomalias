"""Normalización por rangos y combinación de scores."""
from __future__ import annotations

import numpy as np
import pandas as pd


def a_rangos(s: np.ndarray) -> np.ndarray:
    """Convierte scores a rangos en [0, 1] (empates promediados; mayor = más anómalo)."""
    r = pd.Series(s).rank(method="average").to_numpy()
    return (r - 1) / max(len(s) - 1, 1)


def combinar(scores: dict[str, np.ndarray], pesos: dict[str, float] | None = None) -> np.ndarray:
    """Media ponderada de rangos; los pesos se renormalizan sobre los detectores presentes."""
    if not scores:
        raise ValueError("No hay scores que combinar")
    p = {k: (pesos or {}).get(k, 1.0) for k in scores}
    total = sum(p.values())
    return sum(p[k] / total * a_rangos(v) for k, v in scores.items())
