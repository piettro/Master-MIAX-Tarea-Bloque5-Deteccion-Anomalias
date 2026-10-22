"""Orquestación: detectores en paralelo, validación de salida, reintento y degradación."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from anomalias.detectors.base import EspecDetector
from anomalias.domain.ensemble import combinar
from anomalias.domain.leakage_guard import verificar_ajuste, verificar_features
from anomalias.domain.schema import ConfigExperimento, ResultadoDetector
from anomalias.domain.validation import aupr, particiones, precision_en_k
from anomalias.orchestration.trace import Traza


class DetectorObligatorioFallo(RuntimeError):
    """Falló un detector no opcional tras el reintento."""


@dataclass
class Resultado:
    """Scores por detector (solo los que funcionaron) y su traza."""

    scores: dict[str, np.ndarray]
    traza: Traza
    avisos: list[str]


def _ejecutar_uno(
    espec: EspecDetector, seed: int, X_fit: np.ndarray, y_fit: np.ndarray, X_score: np.ndarray
) -> tuple[str, ResultadoDetector | None, list[dict], str | None]:
    """Ajusta, puntúa y valida; reintenta una vez con otra semilla."""
    filas: list[dict] = []
    ultimo_error = ""
    for intento in range(2):
        t0 = time.perf_counter()
        try:
            det = espec.fabrica(seed + intento)
            det.fit(X_fit, y_fit)
            res = ResultadoDetector(
                nombre=espec.nombre, scores=det.score(X_score), n_esperado=len(X_score)
            )
            nota = "ok" if intento == 0 else "ok tras reintento"
            filas.append({"paso": "detector", "modelo": espec.nombre,
                          "segundos": time.perf_counter() - t0, "nota": nota})
            return espec.nombre, res, filas, None
        except Exception as e:  # noqa: BLE001 - se degrada con mensaje claro
            ultimo_error = f"{type(e).__name__}: {e}"
            filas.append({"paso": "detector", "modelo": espec.nombre,
                          "segundos": time.perf_counter() - t0,
                          "nota": f"fallo intento {intento + 1}: {ultimo_error[:80]}"})
    return espec.nombre, None, filas, ultimo_error


def ejecutar_detectores(
    especs: list[EspecDetector],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_score: np.ndarray,
    seed: int = 42,
    n_jobs: int = 2,
    traza: Traza | None = None,
) -> Resultado:
    """Ejecuta los detectores en paralelo (hilos: sklearn libera el GIL).

    Un detector opcional que falla se omite con aviso; uno obligatorio lanza error.
    """
    traza = traza or Traza()
    salidas = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_ejecutar_uno)(e, seed, X_fit, y_fit, X_score) for e in especs
    )
    scores: dict[str, np.ndarray] = {}
    avisos: list[str] = []
    for espec, (nombre, res, filas, error) in zip(especs, salidas):
        for f in filas:
            traza.registrar(f["paso"], f["modelo"], f["segundos"], f["nota"])
        if res is not None:
            scores[nombre] = res.scores
        elif espec.opcional:
            aviso = f"Detector opcional '{nombre}' omitido: {error}"
            warnings.warn(aviso)
            avisos.append(aviso)
        else:
            raise DetectorObligatorioFallo(f"Detector obligatorio '{nombre}' falló: {error}")
    return Resultado(scores, traza, avisos)


def evaluar_cv(
    especs: list[EspecDetector],
    combinaciones: dict[str, tuple[str, ...]],
    X: np.ndarray,
    y: np.ndarray,
    cfg: ConfigExperimento,
    columnas: tuple[str, ...] = (),
    n_jobs: int = 2,
) -> tuple[pd.DataFrame, Traza]:
    """Validación cruzada de detectores y de combinaciones (ensembles) sin fuga.

    Cada detector se ajusta una vez por partición; los ensembles reutilizan sus scores.
    """
    verificar_features(columnas, cfg.columna_objetivo)
    traza, filas = Traza(), []
    pesos = {e.nombre: e.peso for e in especs}
    for i, p in enumerate(particiones(y, cfg)):
        verificar_ajuste(p.idx_fit, p, permitir_produccion=cfg.permitir_produccion)
        res = ejecutar_detectores(
            especs, X[p.idx_fit], y[p.idx_fit], X[p.idx_val], cfg.seed, n_jobs, traza
        )
        candidatos = dict(res.scores)
        for nombre, miembros in combinaciones.items():
            presentes = {m: res.scores[m] for m in miembros if m in res.scores}
            if presentes:
                candidatos[nombre] = combinar(presentes, pesos)
        for nombre, s in candidatos.items():
            filas.append({"modelo": nombre, "particion": i,
                          "aupr": aupr(y[p.idx_val], s), "p_at_k": precision_en_k(y[p.idx_val], s)})
    return pd.DataFrame(filas), traza
