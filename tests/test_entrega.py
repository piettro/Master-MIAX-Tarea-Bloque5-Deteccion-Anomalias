"""Tests de scoring, CSV de entrega y notebook autocontenido (solo sintéticos)."""
import importlib.util
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from anomalias.config import RAIZ
from anomalias.detectors.registro import ENSEMBLE_FINAL, catalogo
from anomalias.domain.leakage_guard import ErrorFuga, verificar_ajuste_no_produccion
from anomalias.domain.validation import deduplicar
from anomalias.orchestration.submission import entrenar_y_puntuar, escribir_scores, validar_csv
from samples.generar_sintetico import generar_sintetico


def _par():
    train = generar_sintetico(1500, seed=1)
    prod = generar_sintetico(500, seed=2)
    prod["es_fraude"] = np.nan
    return deduplicar(train), prod


def _especs():
    cat = catalogo()
    return [cat[n] for n in ENSEMBLE_FINAL]


def test_scoring_produccion_orden_longitud_y_determinismo():
    train, prod = _par()
    s1, res = entrenar_y_puntuar(_especs(), train, prod, "es_fraude", "importe")
    s2, _ = entrenar_y_puntuar(_especs(), train, prod, "es_fraude", "importe")
    assert s1.shape == (len(prod),) and np.isfinite(s1).all() and 0 <= s1.min() and s1.max() <= 1
    assert np.array_equal(s1, s2)
    assert set(res.traza.tabla()["paso"]) == {"features", "detector", "ensemble"}


def test_scoring_no_usa_la_columna_objetivo_de_produccion():
    train, prod = _par()
    s1, _ = entrenar_y_puntuar(_especs(), train, prod, "es_fraude", "importe")
    prod2 = prod.copy()
    prod2["es_fraude"] = 1.0  # cambiar el objetivo de producción no debe alterar el score
    s2, _ = entrenar_y_puntuar(_especs(), train, prod2, "es_fraude", "importe")
    assert np.array_equal(s1, s2)


def test_scoring_detecta_fraude_en_sintetico():
    train, prod = _par()
    real = generar_sintetico(500, seed=2)["es_fraude"].to_numpy()
    s, _ = entrenar_y_puntuar(_especs(), train, prod, "es_fraude", "importe")
    assert s[real == 1].mean() > s[real == 0].mean() + 0.3


def test_ajuste_sobre_produccion_bloqueado():
    X = np.zeros((10, 2))
    with pytest.raises(ErrorFuga):
        verificar_ajuste_no_produccion(X, X)
    with pytest.raises(ErrorFuga):
        verificar_ajuste_no_produccion(X, X[:5])  # comparte memoria
    verificar_ajuste_no_produccion(X, X.copy())


def test_csv_valido_e_invalidos(tmp_path):
    ok = tmp_path / "ok.csv"
    escribir_scores(np.linspace(0, 1, 10), ok)
    assert ok.read_text().splitlines()[0] == "score"
    assert validar_csv(ok, 10) == []
    assert any("Filas" in p for p in validar_csv(ok, 11))
    assert any("No existe" in p for p in validar_csv(tmp_path / "no.csv", 10))
    dos = tmp_path / "dos.csv"
    pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(dos, index=False)
    assert any("1 columna" in p for p in validar_csv(dos, 1))
    nan = tmp_path / "nan.csv"
    pd.DataFrame({"score": [0.1, np.nan, 0.3]}).to_csv(nan, index=False)
    assert any("NaN" in p for p in validar_csv(nan, 3))
    const = tmp_path / "const.csv"
    pd.DataFrame({"score": [0.5] * 4}).to_csv(const, index=False)
    assert any("constante" in p for p in validar_csv(const, 4))


def _constructor():
    spec = importlib.util.spec_from_file_location("build_notebook", RAIZ / "scripts" / "build_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_notebook"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_miembros_vacios_o_placeholder_se_rechazan():
    b = _constructor()
    for malo in ([], ["<nombre>"], ["PENDIENTE"], ["Integrante 1"]):
        with pytest.raises(ValueError):
            b.validar_miembros(malo)
    b.validar_miembros(["Ana Pérez", "Luis García"])


def test_notebook_autocontenido_y_ejecutable_en_demo(tmp_path, monkeypatch):
    b = _constructor()
    nb = b.construir(["Ana Pérez", "Luis García"])
    nbformat.validate(nb)
    codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "import anomalias" not in codigo and "from anomalias" not in codigo
    assert "Ana Pérez" in codigo
    textos = " ".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Reflexión final" in textos and "Integrantes" in textos

    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("OUTPUT_CSV", str(tmp_path / "s.csv"))
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(tmp_path)}}
    )
    assert validar_csv(tmp_path / "s.csv", 2000) == []
