"""Esquema de validación (K-fold estratificado repetido) y métrica AUPR (P-01)."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from anomalias.detectors.base import Detector
from anomalias.domain.leakage_guard import Particion, ajustar_seguro
from anomalias.domain.schema import ConfigExperimento, ResultadoDetector


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas idénticas para que no haya copias entre ajuste y validación."""
    return df.drop_duplicates().reset_index(drop=True)


def aupr(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (equivale a average_precision_score), con empates agrupados."""
    y = np.asarray(y).astype(int)
    orden = np.argsort(-s, kind="stable")
    s_ord, y_ord = s[orden], y[orden]
    tp = np.cumsum(y_ord)
    ultimo = np.r_[np.flatnonzero(np.diff(s_ord)), len(s_ord) - 1]
    tp_u, n_u = tp[ultimo], ultimo + 1
    precision, recall = tp_u / n_u, tp_u / max(y.sum(), 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def precision_en_k(y: np.ndarray, s: np.ndarray, k: int | None = None) -> float:
    """Precisión entre los k scores más altos (k = nº de positivos por defecto)."""
    k = int(k or y.sum())
    return float(np.asarray(y)[np.argsort(-s, kind="stable")[:k]].mean())


def particiones(y: np.ndarray, cfg: ConfigExperimento) -> list[Particion]:
    """K-fold estratificado repetido con semillas fijas."""
    res: list[Particion] = []
    for rep in range(cfg.n_repeats):
        rng = np.random.default_rng(cfg.seed + rep)
        folds = np.empty(len(y), dtype=int)
        for clase in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == clase))
            folds[idx] = np.arange(len(idx)) % cfg.n_splits
        for k in range(cfg.n_splits):
            res.append(Particion(np.flatnonzero(folds != k), np.flatnonzero(folds == k)))
    return res


def evaluar(
    fabrica: Callable[[], Detector],
    X: np.ndarray,
    y: np.ndarray,
    cfg: ConfigExperimento,
    columnas: tuple[str, ...] = (),
) -> pd.DataFrame:
    """AUPR, precisión@k y tiempo por partición; todo ajuste pasa por el guardrail."""
    filas = []
    for i, p in enumerate(particiones(y, cfg)):
        det = fabrica()
        t0 = time.perf_counter()
        ajustar_seguro(det, X, y, p, columnas=columnas, columna_objetivo=cfg.columna_objetivo)
        s = ResultadoDetector(
            nombre=det.nombre, scores=det.score(X[p.idx_val]), n_esperado=len(p.idx_val)
        ).scores
        filas.append(
            {
                "modelo": det.nombre,
                "particion": i,
                "aupr": aupr(y[p.idx_val], s),
                "p_at_k": precision_en_k(y[p.idx_val], s),
                "segundos": time.perf_counter() - t0,
            }
        )
    return pd.DataFrame(filas)
