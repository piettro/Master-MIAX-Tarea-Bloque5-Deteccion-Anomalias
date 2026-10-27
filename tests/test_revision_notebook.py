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


# ---------------------------------------------------------------- comprobación de cifras (no fatal por defecto)
def test_comparar_cifras_tolerancia_y_cota_superior():
    import pandas as pd
    from anomalias.domain.diagnostico import comparar_cifras

    citadas = {"a": (0.834, 0.0005), "b": (0.5, 0.0005), "deriva_max": (0.02, 0)}
    t = comparar_cifras(citadas, {"a": 0.8342, "b": 0.51, "deriva_max": 0.0159})
    assert isinstance(t, pd.DataFrame) and t.set_index("cifra")["ok"].to_dict() == {"a": True, "b": False, "deriva_max": True}
    assert not comparar_cifras({"deriva_max": (0.02, 0)}, {"deriva_max": 0.03})["ok"].iloc[0]


def _espacio_comprobacion(estricta: bool, demo: bool = False) -> dict:
    """Variables mínimas que la celda de comprobación espera, con valores que NO cuadran con lo citado."""
    import pandas as pd

    idx = ["hgb", "rf", "et", "ens_final (hgb+rf+et)", "hgb+rf", "ens_con_gmm", "gmm3"]
    df = pd.DataFrame({"y": [0, 1, 0]})
    return {
        "pd": pd, "USAR_DEMO": demo, "SEED": 42, "CV_REPETICIONES": 1, "COMPROBACION_ESTRICTA": estricta,
        "tabla_cv": pd.DataFrame({"AUPR": 0.5, "desv_entre_folds": 0.5}, index=idx),
        "df_const_bruto": df, "df_prod": df, "df_const": df, "pos": 1, "OBJETIVO": "y", "dif": pd.Series([0.5]),
        "BASE_GMM5": 0.1,
        "evid_features": pd.DataFrame({"AUPR GMM(3)": 0.1, "AUPR HGB": 0.1}, index=["crudas", "transformadas"]),
        "evid_reg": pd.DataFrame({"AUPR": [0.1, 0.2]}),
        "SUBGRUPO": {"n_fuerte": 1, "n_suave": 1, "recall_fuerte": 0.1, "recall_suave": 0.1, "frac_suave_no_detectado": 0.1},
    }


def _celda_comprobacion() -> str:
    return _constructor().c_comprobacion()


def test_celda_comprobacion_avisa_y_continua_por_defecto(capsys):
    from anomalias.domain.diagnostico import comparar_cifras

    ns = _espacio_comprobacion(estricta=False)
    ns["comparar_cifras"] = comparar_cifras
    exec(_celda_comprobacion(), ns)  # no debe lanzar
    salida = capsys.readouterr().out
    assert "AVISO" in salida and "COMPROBACION_ESTRICTA=true" in salida


def test_celda_comprobacion_estricta_detiene_el_notebook():
    from anomalias.domain.diagnostico import comparar_cifras

    ns = _espacio_comprobacion(estricta=True)
    ns["comparar_cifras"] = comparar_cifras
    with pytest.raises(AssertionError, match="no cuadran"):
        exec(_celda_comprobacion(), ns)


def test_celda_comprobacion_se_omite_en_demo(capsys):
    ns = _espacio_comprobacion(estricta=True, demo=True)
    exec(_celda_comprobacion(), ns)
    assert "omitida en demo" in capsys.readouterr().out


def test_notebook_define_la_variable_de_comprobacion_estricta():
    nb = _constructor().construir(["Ana Pérez", "Luis García"])
    codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert 'COMPROBACION_ESTRICTA = os.environ.get("COMPROBACION_ESTRICTA"' in codigo
    assert "def comparar_cifras(" in codigo and "from anomalias" not in codigo
