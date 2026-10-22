"""Tests de métrica, particiones y guardrail anti-fuga (solo sintéticos)."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from anomalias.detectors.mock import MockDetector
from anomalias.domain.leakage_guard import (
    ErrorFuga, Particion, ajustar_seguro, estadisticas_de_ajuste,
    verificar_ajuste, verificar_features,
)
from anomalias.domain.schema import ConfigExperimento, ResultadoDetector
from anomalias.domain.validation import aupr, evaluar, particiones
from samples.generar_sintetico import generar_sintetico


def _datos():
    df = generar_sintetico(1500)
    return df.drop(columns="es_fraude").to_numpy(float), df["es_fraude"].to_numpy()


def test_aupr_igual_a_sklearn_con_empates():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.05).astype(int)
    s = rng.integers(0, 20, 500).astype(float)
    assert aupr(y, s) == pytest.approx(average_precision_score(y, s))


def test_particiones_disjuntas_estratificadas_y_deterministas():
    _, y = _datos()
    cfg = ConfigExperimento(n_splits=5, n_repeats=2)
    ps = particiones(y, cfg)
    assert len(ps) == 10
    for p in ps:
        assert np.intersect1d(p.idx_fit, p.idx_val).size == 0
        assert len(p.idx_fit) + len(p.idx_val) == len(y)
        assert abs(y[p.idx_val].sum() - y.sum() / 5) <= 1
    assert all(np.array_equal(a.idx_val, b.idx_val) for a, b in zip(ps, particiones(y, cfg)))


def test_particion_solapada_falla():
    with pytest.raises(ErrorFuga):
        Particion(np.array([1, 2, 3]), np.array([3, 4]))


def test_objetivo_como_feature_falla():
    with pytest.raises(ErrorFuga):
        verificar_features(["a", "Class"], "Class")


def test_ajuste_con_validacion_falla():
    p = Particion(np.arange(80), np.arange(80, 100))
    with pytest.raises(ErrorFuga):
        verificar_ajuste(np.arange(90), p)


def test_ajuste_sobre_produccion_requiere_autorizacion():
    p = Particion(np.arange(80), np.arange(80, 100))
    with pytest.raises(ErrorFuga):
        verificar_ajuste(p.idx_fit, p, es_produccion=True)
    verificar_ajuste(p.idx_fit, p, es_produccion=True, permitir_produccion=True)


def test_detector_solo_ve_filas_de_ajuste():
    X, y = _datos()
    vistas = {}

    class Espia(MockDetector):
        def fit(self, X_, y_=None):
            vistas["n"] = len(X_)
            return super().fit(X_, y_)

    p = particiones(y, ConfigExperimento())[0]
    ajustar_seguro(Espia(), X, y, p)
    assert vistas["n"] == len(p.idx_fit)


def test_estadisticas_solo_con_ajuste():
    X, y = _datos()
    p = particiones(y, ConfigExperimento())[0]
    media, _ = estadisticas_de_ajuste(X, p)
    assert np.allclose(media, X[p.idx_fit].mean(axis=0))
    assert not np.allclose(media, X.mean(axis=0))


def test_evaluar_con_mock_y_objetivo_en_columnas():
    X, y = _datos()
    res = evaluar(MockDetector, X, y, ConfigExperimento(n_repeats=1))
    assert len(res) == 5 and (res["aupr"] > y.mean()).all()
    with pytest.raises(ErrorFuga):
        evaluar(MockDetector, X, y, ConfigExperimento(), columnas=("importe", "Class"))


def test_resultado_detector_rechaza_nan_y_longitud():
    with pytest.raises(ValueError):
        ResultadoDetector(nombre="x", scores=np.array([1.0, np.nan]), n_esperado=2)
    with pytest.raises(ValueError):
        ResultadoDetector(nombre="x", scores=np.array([1.0]), n_esperado=2)
