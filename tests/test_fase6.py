"""Tests de diagnósticos de la Fase 6 y del envoltorio de variables derivadas (solo sintéticos)."""
import numpy as np
import pytest

from anomalias.detectors.aumentado import AumentoDerivadas
from anomalias.detectors.mock import MockDetector
from anomalias.domain.diagnostico import (
    comparacion_pareada, fraccion_con_copia, importancia_permutacion, recall_en_k,
)


class _SoloColumna0:
    nombre = "col0"

    def fit(self, X, y=None):
        return self

    def score(self, X):
        return X[:, 0]


def _datos(n=4000, seed=0):
    """3 variables + importe (log1p). Fraude: cola inferior en la col. 0 y el importe 149,98 repetido."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    imp = np.round(rng.lognormal(3, 1, n), 2)
    y = np.zeros(n, int)
    y[rng.choice(n, 80, replace=False)] = 1
    X[y == 1, 0] = -5 + rng.normal(0, .2, 80)
    imp[y == 1] = 149.98
    return np.column_stack([X, np.log1p(imp)]), y


def test_importancia_permutacion_detecta_la_columna_usada():
    X, y = _datos()
    imp = importancia_permutacion(_SoloColumna0(), -X, y, n_rep=2)  # score alto = fraude → -col0
    assert imp[0] > 0.3 and np.abs(imp[1:]).max() < 1e-9


def test_comparacion_pareada():
    r = comparacion_pareada(np.array([.5, .6, .7, .8]), np.array([.4, .5, .6, .7]))
    assert r["dif_media"] == pytest.approx(0.1) and r["gana_en"] == 4 and r["particiones"] == 4


def test_recall_en_k_global_y_por_grupo():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([.9, .8, .1, .7, .2, .3])
    assert recall_en_k(y, s) == pytest.approx(2 / 3)
    assert recall_en_k(y, s, grupo=np.array([1, 0, 0, 0, 0, 0])) == 1.0


def test_fraccion_con_copia():
    a = np.array([[1., 2.], [3., 4.]])
    assert fraccion_con_copia(a, np.array([[1., 2.], [9., 9.]])) == 0.5


def test_aumento_aprende_solo_del_ajuste_y_detecta_cola_e_importe():
    X, y = _datos()
    fit, val = np.arange(3000), np.arange(3000, 4000)
    m = AumentoDerivadas(MockDetector(), col_importe=3, colas=True, importes=True, frecuencia=True).fit(X[fit], y[fit])
    assert 149.98 in set(m._clave.tolist())
    assert m._vars == [0] and m._lados == [-1]
    extra = m._extras(X[val])
    assert extra.shape == (1000, 3)
    assert extra[y[val] == 1, 0].mean() > extra[y[val] == 0, 0].mean()  # más colas tocadas en fraude
    assert extra[y[val] == 1, 1].mean() > .9                            # importe clave reconocido


def test_importe_clave_solo_en_validacion_no_se_aprende():
    X, y = _datos()
    X = X.copy()
    fit, val = np.arange(3000), np.arange(3000, 4000)
    y2 = y.copy()
    y2[fit] = 0  # en ajuste no hay fraudes con ese importe
    m = AumentoDerivadas(MockDetector(), col_importe=3, importes=True)
    m.fit(X[fit], np.r_[1, np.zeros(2999, int)])
    assert 149.98 not in set(m._clave.tolist())


def test_aumento_es_determinista_y_cumple_protocol():
    from anomalias.detectors.base import Detector
    X, y = _datos()
    a = AumentoDerivadas(MockDetector(), 3, colas=True).fit(X[:3000], y[:3000]).score(X[3000:])
    b = AumentoDerivadas(MockDetector(), 3, colas=True).fit(X[:3000], y[:3000]).score(X[3000:])
    assert np.array_equal(a, b) and isinstance(AumentoDerivadas(MockDetector(), 3), Detector)
