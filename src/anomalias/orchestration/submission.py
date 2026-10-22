"""Scoring de producción, escritura del CSV (R-06) y su validación."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalias.detectors.base import EspecDetector
from anomalias.domain.ensemble import combinar
from anomalias.domain.features import construir_features
from anomalias.domain.leakage_guard import verificar_ajuste_no_produccion
from anomalias.orchestration.pipeline import Resultado, ejecutar_detectores
from anomalias.orchestration.trace import Traza

COLUMNA_SCORE = "score"  # P-04: cabecera única, sin índice


def entrenar_y_puntuar(
    especs: list[EspecDetector],
    df_train: pd.DataFrame,
    df_prod: pd.DataFrame,
    objetivo: str,
    col_importe: str | None,
    seed: int = 42,
    n_jobs: int = 2,
) -> tuple[np.ndarray, Resultado]:
    """Ajusta con construcción y puntúa producción; devuelve el score del ensemble."""
    traza = Traza()
    with traza.medir("features"):
        X_fit, cols = construir_features(df_train, objetivo, col_importe)
        X_prod, cols_prod = construir_features(df_prod, objetivo, col_importe)
    if cols != cols_prod:
        raise ValueError(f"Columnas distintas entre construcción y producción: {cols} / {cols_prod}")
    verificar_ajuste_no_produccion(X_fit, X_prod)
    y_fit = df_train[objetivo].to_numpy(int)
    res = ejecutar_detectores(especs, X_fit, y_fit, X_prod, seed, n_jobs, traza)
    with traza.medir("ensemble", "+".join(res.scores), "media de rangos"):
        final = combinar(res.scores, {e.nombre: e.peso for e in especs})
    return final, res


def escribir_scores(scores: np.ndarray, ruta: str | Path) -> None:
    """CSV de una sola columna con cabecera y sin índice."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({COLUMNA_SCORE: scores}).to_csv(ruta, index=False)


def validar_csv(ruta: str | Path, n_filas: int) -> list[str]:
    """Devuelve la lista de problemas del CSV de entrega (vacía = válido)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return [f"No existe {ruta}"]
    df = pd.read_csv(ruta)
    problemas: list[str] = []
    if df.shape[1] != 1:
        problemas.append(f"Debe tener 1 columna y tiene {df.shape[1]}")
        return problemas
    if df.columns[0] != COLUMNA_SCORE:
        problemas.append(f"Cabecera '{df.columns[0]}' != '{COLUMNA_SCORE}'")
    if len(df) != n_filas:
        problemas.append(f"Filas {len(df)} != {n_filas} del fichero de producción")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    if not np.isfinite(s).all():
        problemas.append("Hay valores no numéricos, NaN o infinitos")
    elif np.ptp(s) == 0:
        problemas.append("Score constante: no discrimina")
    return problemas
