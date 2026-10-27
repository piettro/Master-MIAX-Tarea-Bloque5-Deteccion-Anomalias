"""Tests de evaluar_cv con oof y de la celda de configuración del notebook (sin red)."""
import zipfile
from urllib.error import URLError

import numpy as np
import pytest

from anomalias.detectors.base import EspecDetector
from anomalias.detectors.mock import MockDetector
from anomalias.domain.features import construir_features
from anomalias.domain.schema import ConfigExperimento
from anomalias.orchestration.pipeline import evaluar_cv
from samples.generar_sintetico import generar_sintetico
from tests.test_entrega import _constructor


def _mock(s):
    return MockDetector()


def _cv_datos():
    df = generar_sintetico(600, seed=3)
    X, cols = construir_features(df, "es_fraude", col_importe="importe")
    especs = [EspecDetector("a", _mock), EspecDetector("b", _mock)]
    return especs, X, df["es_fraude"].to_numpy(), cols


def test_evaluar_cv_oof_no_altera_resultado_y_cubre_cada_fila_una_vez():
    especs, X, y, cols = _cv_datos()
    cfg = ConfigExperimento(n_repeats=2)
    comb = {"ens": ["a", "b"]}
    sin, _ = evaluar_cv(especs, comb, X, y, cfg, cols)
    oof: dict[str, np.ndarray] = {}
    con, _ = evaluar_cv(especs, comb, X, y, cfg, cols, oof=oof)
    assert sin.equals(con)
    assert set(oof) == {"a", "b", "ens"}
    for s in oof.values():
        assert s.shape == (len(y),) and not np.isnan(s).any()


def test_evaluar_cv_oof_se_vacia_al_reutilizarlo():
    especs, X, y, cols = _cv_datos()
    cfg = ConfigExperimento(n_repeats=1)
    oof = {"viejo": np.zeros(3)}
    evaluar_cv(especs, {}, X, y, cfg, cols, oof=oof)
    assert "viejo" not in oof and set(oof) == {"a", "b"}


def _ejecutar_config(tmp_path, monkeypatch, demo=None):
    b = _constructor()
    monkeypatch.chdir(tmp_path)
    if demo is None:
        monkeypatch.delenv("DEMO_MODE", raising=False)
    else:
        monkeypatch.setenv("DEMO_MODE", demo)
    monkeypatch.delenv("DESCARGAR_DATOS", raising=False)
    ns: dict = {}
    exec(b.C_IMPORTS, ns)
    exec(b.C_CONFIG, ns)
    return ns


def test_config_sin_datos_ni_demo_lanza_filenotfound_con_nombre(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="credit_card_construction.csv"):
        _ejecutar_config(tmp_path, monkeypatch)


def test_config_localiza_zips_en_data(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    for n in ("construction", "production"):
        (tmp_path / "data" / f"credit_card_{n}.csv.zip").write_bytes(b"x")
    ns = _ejecutar_config(tmp_path, monkeypatch)
    assert ns["USAR_DEMO"] is False
    assert ns["RUTA_CONSTRUCCION"].name == "credit_card_construction.csv.zip"
    assert ns["RUTA_PRODUCCION"].name == "credit_card_production.csv.zip"


def test_config_demo_mode_usa_demo_sin_datos(tmp_path, monkeypatch):
    ns = _ejecutar_config(tmp_path, monkeypatch, demo="true")
    assert ns["USAR_DEMO"] is True


def _ns_descarga(tmp_path, monkeypatch):
    return _ejecutar_config(tmp_path, monkeypatch, demo="true")


def test_descargar_zip_valido_se_renombra(tmp_path, monkeypatch):
    ns = _ns_descarga(tmp_path, monkeypatch)

    def falso(url, destino):
        with zipfile.ZipFile(destino, "w") as z:
            z.writestr("a.csv", "x\n1\n")

    ns["urlretrieve"] = falso
    destino = tmp_path / "d.csv.zip"
    assert ns["descargar_zip"]("http://x", destino) == destino
    assert zipfile.is_zipfile(destino) and not (tmp_path / "d.csv.zip.tmp").exists()


def test_descargar_zip_no_zip_lanza_y_limpia(tmp_path, monkeypatch):
    ns = _ns_descarga(tmp_path, monkeypatch)
    ns["urlretrieve"] = lambda url, destino: destino.write_text("<html>")
    destino = tmp_path / "d.csv.zip"
    with pytest.raises(RuntimeError, match="zip"):
        ns["descargar_zip"]("http://x", destino)
    assert not destino.exists() and not (tmp_path / "d.csv.zip.tmp").exists()


def test_descargar_zip_urlerror_lanza_runtimeerror(tmp_path, monkeypatch):
    ns = _ns_descarga(tmp_path, monkeypatch)

    def falla(url, destino):
        raise URLError("sin red")

    ns["urlretrieve"] = falla
    destino = tmp_path / "d.csv.zip"
    with pytest.raises(RuntimeError, match="red"):
        ns["descargar_zip"]("http://x", destino)
    assert not destino.exists()
