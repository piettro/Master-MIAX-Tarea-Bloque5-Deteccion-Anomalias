"""Tests de features, ensemble, orquestación paralela, degradación y determinismo."""
import warnings

import numpy as np
import pytest

from anomalias.detectors.base import EspecDetector
from anomalias.detectors.mock import MockDetector
from anomalias.detectors.registro import ENSEMBLE_FINAL, catalogo
from anomalias.domain.ensemble import a_rangos, combinar
from anomalias.domain.features import construir_features
from anomalias.domain.leakage_guard import ErrorFuga
from anomalias.domain.schema import ConfigExperimento
from anomalias.orchestration.pipeline import (
    DetectorObligatorioFallo, ejecutar_detectores, evaluar_cv,
)
from samples.generar_sintetico import generar_sintetico


def _datos():
    df = generar_sintetico(1200)
    X, cols = construir_features(df, "es_fraude", col_importe="importe")
    return df, X, df["es_fraude"].to_numpy(), cols


def test_features_sin_objetivo_finitas_y_sin_mutar():
    df, X, _, cols = _datos()
    copia = df.copy()
    assert "es_fraude" not in cols and X.shape == (1200, 4)
    assert np.isfinite(X).all() and df.equals(copia)


def test_rangos_en_01_y_empates():
    r = a_rangos(np.array([5.0, 1.0, 2.0, 9.0]))
    assert r.min() == 0 and r.max() == 1 and r[3] > r[0] > r[2] > r[1]
    e = a_rangos(np.array([5.0, 1.0, 1.0, 9.0]))
    assert e[1] == e[2] and 0 <= e.min() and e.max() == 1


def test_combinar_renormaliza_pesos_y_conserva_orden():
    a, b = np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40])
    c = combinar({"a": a, "b": b}, {"a": 1, "b": 3})
    assert np.array_equal(np.argsort(c), np.argsort(a))
    assert combinar({"a": a}).shape == (4,)


class _Roto:
    nombre = "roto"

    def __init__(self, tipo="excepcion"):
        self.tipo = tipo

    def fit(self, X, y=None):
        if self.tipo == "excepcion":
            raise RuntimeError("boom")
        return self

    def score(self, X):
        return np.full(len(X), np.nan)


def _mock(s):
    return MockDetector()


def test_degradacion_detector_opcional():
    _, X, y, _ = _datos()
    especs = [
        EspecDetector("mock", _mock, opcional=False),
        EspecDetector("roto", lambda s: _Roto(), opcional=True),
        EspecDetector("nan", lambda s: _Roto("nan"), opcional=True),
    ]
    with pytest.warns(UserWarning, match="omitido"):
        res = ejecutar_detectores(especs, X, y, X)
    assert set(res.scores) == {"mock"} and len(res.avisos) == 2
    assert res.traza.tabla().query("modelo == 'roto'").shape[0] == 2  # intento + reintento


def test_detector_obligatorio_que_falla_lanza():
    _, X, y, _ = _datos()
    with pytest.raises(DetectorObligatorioFallo):
        ejecutar_detectores([EspecDetector("roto", lambda s: _Roto(), opcional=False)], X, y, X)


def test_reintento_con_otra_semilla_recupera():
    _, X, y, _ = _datos()

    class Flaky(MockDetector):
        def __init__(self, seed):
            super().__init__()
            self.seed = seed

        def score(self, X):
            s = super().score(X)
            return np.full(len(X), np.nan) if self.seed == 42 else s

    res = ejecutar_detectores([EspecDetector("f", Flaky, opcional=False)], X, y, X, seed=42)
    assert "f" in res.scores
    assert "reintento" in res.traza.tabla()["nota"].iloc[-1]


def test_traza_tiene_columnas_y_es_paralelo_determinista():
    _, X, y, _ = _datos()
    especs = [EspecDetector(n, _mock) for n in ("a", "b", "c")]
    r1 = ejecutar_detectores(especs, X, y, X, n_jobs=3)
    r2 = ejecutar_detectores(especs, X, y, X, n_jobs=1)
    assert list(r1.traza.tabla().columns) == ["paso", "modelo", "segundos", "nota"]
    assert all(np.array_equal(r1.scores[k], r2.scores[k]) for k in "abc")


def test_evaluar_cv_ensemble_supera_azar_y_es_reproducible():
    _, X, y, cols = _datos()
    especs = [catalogo()[n] for n in ENSEMBLE_FINAL]
    cfg = ConfigExperimento(n_repeats=1)
    res1, traza = evaluar_cv(especs, {"ens": ENSEMBLE_FINAL}, X, y, cfg, cols)
    res2, _ = evaluar_cv(especs, {"ens": ENSEMBLE_FINAL}, X, y, cfg, cols)
    medias = res1.groupby("modelo").aupr.mean()
    assert medias["ens"] > 10 * y.mean()
    assert res1["aupr"].tolist() == res2["aupr"].tolist()
    assert len(traza.tabla()) == 5 * 3


def test_evaluar_cv_rechaza_objetivo_como_columna():
    _, X, y, _ = _datos()
    with pytest.raises(ErrorFuga):
        evaluar_cv([EspecDetector("m", _mock)], {}, X, y, ConfigExperimento(), ("a", "Class"))
