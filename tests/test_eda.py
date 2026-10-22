"""Estructura del notebook de EDA (sin datos reales: solo se construye y valida)."""
import importlib.util
import sys

import nbformat

from anomalias.config import RAIZ


def _eda():
    spec = importlib.util.spec_from_file_location("build_eda", RAIZ / "scripts" / "build_eda.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_eda"] = mod
    spec.loader.exec_module(mod)
    return mod.construir()


def test_eda_valido_autocontenido_y_con_secciones():
    nb = _eda()
    nbformat.validate(nb)
    codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    texto = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "import anomalias" not in codigo and "from anomalias" not in codigo
    for n in range(1, 14):
        assert f"## {n}. " in texto or f'"## {n}. ' in codigo, f"falta la sección {n}"
    assert "Hipótesis" in texto and "Mapa hallazgo" in codigo


def test_eda_no_incrusta_datos_en_las_celdas():
    nb = _eda()
    assert not any(".head()" in c.source for c in nb.cells if c.cell_type == "code")


def test_eda_incluye_verificacion_automatica_y_hipotesis_refutables():
    nb = _eda()
    codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    texto = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "assert fallos.empty" in codigo, "la verificación debe detener el notebook si una afirmación falla"
    assert "12b. Verificación automática" in texto
    for h in range(1, 10):
        assert f"| H{h} |" in texto
    assert "Se refuta si" in texto


def test_eda_contiene_los_analisis_de_la_fase_5():
    nb = _eda()
    codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for clave in ("firma", "adjusted_rand_score", "Hanley-McNeil", "kstest", "frec"):
        assert clave in codigo, f"falta el análisis: {clave}"
