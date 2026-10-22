"""Tests de humo (solo datos sintéticos)."""
import ast
from pathlib import Path

import numpy as np

from anomalias.config import RAIZ, cargar_config
from anomalias.detectors.base import Detector
from anomalias.detectors.mock import MockDetector
from samples.generar_sintetico import generar_sintetico

LIBS_MODELADO = {"sklearn", "lightgbm", "xgboost", "torch", "tensorflow", "scipy"}


def test_sintetico_forma_y_determinismo():
    a, b = generar_sintetico(500, seed=1), generar_sintetico(500, seed=1)
    assert a.shape == (500, 5) and a.equals(b)
    assert a["es_fraude"].sum() == 10


def test_mock_cumple_protocol_y_detecta():
    df = generar_sintetico(1000)
    X, y = df.drop(columns="es_fraude").to_numpy(float), df["es_fraude"].to_numpy()
    det = MockDetector()
    assert isinstance(det, Detector)
    s = det.fit(X).score(X)
    assert s.shape == (1000,) and np.isfinite(s).all()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_config_demo_si_faltan_datos(monkeypatch):
    monkeypatch.setenv("TRAIN_FILE", "data/no_existe.csv")
    assert cargar_config().usar_demo


def test_libs_de_modelado_solo_en_detectors():
    """Regla 2: ningún import de modelado fuera de detectors/."""
    for ruta in (RAIZ / "src" / "anomalias").rglob("*.py"):
        if "detectors" in ruta.parts:
            continue
        for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(nodo, ast.Import):
                mods = [a.name for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                mods = [nodo.module]
            assert not {m.split(".")[0] for m in mods} & LIBS_MODELADO, ruta
