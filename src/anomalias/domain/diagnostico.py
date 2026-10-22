"""Diagnósticos de validación: importancia por permutación, comparación pareada y cobertura por grupo."""
from __future__ import annotations

import numpy as np

from anomalias.detectors.base import Detector
from anomalias.domain.validation import aupr


def importancia_permutacion(
    detector: Detector, X_val: np.ndarray, y_val: np.ndarray, n_rep: int = 3, seed: int = 42
) -> np.ndarray:
    """Caída media del AUPR al permutar cada columna de validación (detector ya ajustado)."""
    rng = np.random.default_rng(seed)
    base = aupr(y_val, detector.score(X_val))
    caida = np.zeros(X_val.shape[1])
    for j in range(X_val.shape[1]):
        for _ in range(n_rep):
            Xp = X_val.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            caida[j] += base - aupr(y_val, detector.score(Xp))
    return caida / n_rep


def comparacion_pareada(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Diferencia a - b por partición: media, error típico, z aproximado y nº de particiones donde a gana."""
    d = np.asarray(a, float) - np.asarray(b, float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {
        "dif_media": float(d.mean()),
        "error_tipico": float(se),
        "z": float(d.mean() / se) if se and se > 0 else float("nan"),
        "gana_en": int((d > 0).sum()),
        "particiones": int(len(d)),
    }


def recall_en_k(y: np.ndarray, s: np.ndarray, grupo: np.ndarray | None = None, k: int | None = None) -> float:
    """Fracción de positivos (del grupo, si se indica) que caen entre los k scores más altos (k = nº de positivos)."""
    y = np.asarray(y).astype(int)
    k = int(k or y.sum())
    top = np.zeros(len(y), bool)
    top[np.argsort(-s, kind="stable")[:k]] = True
    sel = (y == 1) if grupo is None else (y == 1) & np.asarray(grupo, bool)
    return float(top[sel].mean()) if sel.any() else float("nan")


def fraccion_con_copia(X_ajuste: np.ndarray, X_val: np.ndarray) -> float:
    """Fracción de filas de validación con una copia exacta en las filas de ajuste."""
    vistas = {fila.tobytes() for fila in np.ascontiguousarray(X_ajuste)}
    return float(np.mean([fila.tobytes() in vistas for fila in np.ascontiguousarray(X_val)]))
