"""Guardrail determinista anti-fuga: lanza ErrorFuga ante cualquier violación."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from anomalias.detectors.base import Detector


class ErrorFuga(AssertionError):
    """Se intentó una operación que filtraría información de validación o producción."""


@dataclass(frozen=True)
class Particion:
    """Índices de ajuste y de validación, disjuntos por construcción."""

    idx_fit: np.ndarray
    idx_val: np.ndarray

    def __post_init__(self) -> None:
        if np.intersect1d(self.idx_fit, self.idx_val).size:
            raise ErrorFuga("Las filas de ajuste y validación se solapan")


def verificar_features(columnas: Sequence[str], columna_objetivo: str) -> None:
    """La columna objetivo no puede ser variable de entrada."""
    if columna_objetivo in columnas:
        raise ErrorFuga(f"La columna objetivo '{columna_objetivo}' está entre las variables")


def verificar_ajuste(
    idx_fit: np.ndarray,
    particion: Particion,
    *,
    es_produccion: bool = False,
    permitir_produccion: bool = False,
) -> None:
    """Ajustar solo con filas de ajuste; producción solo si P-05 lo autoriza."""
    if es_produccion and not permitir_produccion:
        raise ErrorFuga("Ajuste sobre producción no autorizado (P-05)")
    if np.intersect1d(idx_fit, particion.idx_val).size:
        raise ErrorFuga("El ajuste usa filas de validación")
    if not np.isin(idx_fit, particion.idx_fit).all():
        raise ErrorFuga("El ajuste usa filas fuera de la partición de ajuste")


def verificar_ajuste_no_produccion(
    X_fit: np.ndarray, X_prod: np.ndarray, permitir_produccion: bool = False
) -> None:
    """P-05: los datos de ajuste no pueden ser (ni compartir memoria con) los de producción."""
    if not permitir_produccion and (X_fit is X_prod or np.shares_memory(X_fit, X_prod)):
        raise ErrorFuga("Los datos de ajuste son los de producción (P-05 no autoriza)")


def estadisticas_de_ajuste(X: np.ndarray, particion: Particion) -> tuple[np.ndarray, np.ndarray]:
    """Vía sancionada para media/desv.: se calculan tras el split y solo con idx_fit."""
    base = X[particion.idx_fit]
    return base.mean(axis=0), base.std(axis=0) + 1e-9


def ajustar_seguro(
    detector: Detector,
    X: np.ndarray,
    y: np.ndarray | None,
    particion: Particion,
    *,
    columnas: Sequence[str] = (),
    columna_objetivo: str = "Class",
) -> Detector:
    """Ajusta el detector solo con X[idx_fit]; verifica antes columnas e índices."""
    verificar_features(columnas, columna_objetivo)
    verificar_ajuste(particion.idx_fit, particion)
    y_fit = None if y is None else y[particion.idx_fit]
    return detector.fit(X[particion.idx_fit], y_fit)
