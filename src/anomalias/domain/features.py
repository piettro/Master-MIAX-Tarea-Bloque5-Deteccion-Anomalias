"""Ingeniería de variables pura (sin I/O ni estadísticas aprendidas: no puede haber fuga)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from anomalias.domain.leakage_guard import verificar_features

ESCALA = 100.0  # constante fija, no aprendida de los datos


def construir_features(
    df: pd.DataFrame, columna_objetivo: str, col_importe: str | None = "Amount"
) -> tuple[np.ndarray, tuple[str, ...]]:
    """arcsinh(x/ESCALA) en las variables (colas pesadas) y log1p en el importe."""
    cols = tuple(c for c in df.columns if c != columna_objetivo)
    verificar_features(cols, columna_objetivo)
    salida = []
    for c in cols:
        v = df[c].to_numpy(float)
        salida.append(np.log1p(np.clip(v, 0, None)) if c == col_importe else np.arcsinh(v / ESCALA))
    return np.column_stack(salida), cols
