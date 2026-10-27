"""Ensambla entrega/taller_anomalias.ipynb autocontenido: inlinea el código de src/ en celdas."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from anomalias.config import cargar_config  # noqa: E402

PLACEHOLDERS = ("pendiente", "nombre", "integrante", "<", "xxx", "todo")

# Única fuente de las cifras citadas en markdown: clave -> (valor, tolerancia, decimales).
# La celda de comprobación del notebook las contrasta con lo calculado en la ejecución.
CIFRAS: dict[str, tuple[float, float, int]] = {
    "n_const": (170884, 0, 0), "n_prod": (113923, 0, 0), "n_fraudes": (295, 0, 0),
    "n_dedup": (166717, 0, 0), "n_fraudes_dedup": (285, 0, 0),
    "deriva_max": (0.02, 0, 2),  # cota superior, no valor exacto
    "aupr_ens": (0.834, 0.0005, 3), "aupr_hgb": (0.834, 0.0005, 3), "aupr_hgb_rf": (0.834, 0.0005, 3),
    "aupr_rf": (0.823, 0.0005, 3), "aupr_et": (0.820, 0.0005, 3),
    "aupr_ens_gmm": (0.806, 0.0005, 3), "aupr_gmm3": (0.706, 0.0005, 3),
    "gan_ens_hgb": (0.0006, 0.00005, 4),
    "desv_min": (0.055, 0.0005, 3), "desv_max": (0.069, 0.0005, 3),
    "aupr_base_gmm5": (0.375, 0.0005, 3),
    "evid_gmm_crudo": (0.750, 0.0005, 3), "evid_gmm_trans": (0.732, 0.0005, 3),
    "evid_hgb_crudo": (0.836, 0.0005, 3), "evid_hgb_trans": (0.836, 0.0005, 3),
    "hgb_defecto": (0.002, 0.0005, 3), "hgb_reg": (0.836, 0.0005, 3),
    "n_fuerte": (218, 0, 0), "n_suave": (67, 0, 0),
    "recall_fuerte": (0.996, 0.0005, 3), "recall_suave": (0.310, 0.0005, 3),
    "frac_suave_no_detectado": (0.979, 0.0005, 3),
}


def _es(valor: float, dec: int) -> str:
    """Formato español: coma decimal y espacio como separador de miles."""
    t = f"{valor:,.{dec}f}"
    return t.replace(",", " ").replace(".", ",")


TXT = {k: _es(v, d) for k, (v, _, d) in CIFRAS.items()}


def validar_miembros(miembros: list[str]) -> None:
    """R-02: exige nombres reales (ni vacío ni marcadores de posición)."""
    if not miembros:
        raise ValueError("Faltan los integrantes: defínelos en MIEMBROS (.env) o con --miembros")
    for m in miembros:
        if any(p in m.lower() for p in PLACEHOLDERS):
            raise ValueError(f"Integrante con aspecto de marcador de posición: {m!r}")


def fuente_inline(rutas: list[str]) -> str:
    """Concatena módulos de src/ sin los imports del propio paquete ni __future__."""
    partes = []
    for rel in rutas:
        ruta = RAIZ / rel
        src = ruta.read_text(encoding="utf-8")
        lineas, borrar = src.splitlines(), set()
        for n in ast.parse(src).body:
            propio = isinstance(n, ast.ImportFrom) and (
                n.module == "__future__" or (n.module or "").startswith("anomalias")
            )
            if propio:
                borrar.update(range(n.lineno - 1, n.end_lineno))
        cuerpo = "\n".join(l for i, l in enumerate(lineas) if i not in borrar).strip()
        partes.append(f"# ---- {rel} ----\n{cuerpo}\n")
    return "\n\n".join(partes)


MD_RESUMEN = """## 2. Resumen ejecutivo

**Problema.** Asignar un score de fraude a cada fila de `credit_card_production.csv` ({n_prod} transacciones). La métrica es el **AUPR** (`average_precision_score`) y mayor score = más probable fraude, según el notebook de partida.

**Datos.** El fichero de construcción tiene {n_const} filas con etiqueta ({n_fraudes} fraudes, 0,17 %). Producción tiene las mismas 21 variables numéricas, sin etiqueta, y una distribución prácticamente idéntica a la de construcción (diferencia de medias tipificada por variable < {deriva_max}; ver la auditoría de la sección 3).

**Enfoque.** Aprendizaje **supervisado** con tres modelos de árboles (gradient boosting regularizado, Random Forest y Extra Trees) combinados por **media de rangos**. Los modelos no supervisados (mezcla gaussiana, Isolation Forest) se evaluaron y se descartaron porque rinden menos y no mejoran el ensemble.

**Resultado de validación** (5 folds x 1 repetición, sin fuga, semilla 42): ensemble **AUPR = {aupr_ens}**, frente a {aupr_base_gmm5} del modelo del notebook de partida (GMM de 5 componentes sobre variables crudas), recalculado aquí con las mismas particiones. La desviación entre folds va de {desv_min} a {desv_max} (hay ~57 fraudes por fold), por lo que las diferencias de unas milésimas entre los modelos supervisados **no son significativas**.

**Cómo leer este notebook.** Cada decisión (variables, validación, hiperparámetros, ensemble) va con una celda que la justifica con cifras calculadas en la propia ejecución. Las cifras citadas en el texto se contrastan automáticamente con lo calculado en la celda de comprobación de la sección 9."""

MD_MAPA = """### Mapa del notebook

| Sección | Qué contiene |
|---|---|
| 1. Integrantes | Miembros del grupo |
| 2. Resumen ejecutivo | Problema, datos, enfoque y resultado |
| Entorno y configuración | Librerías, variables de entorno, localización de los datos y código de apoyo (plegado) |
| 3. Carga y auditoría | Forma, nulos, duplicados, desbalanceo y deriva producción vs. construcción |
| 4. Ingeniería de variables | Transformación fija de variables y su evidencia |
| 5. Validación y control de fuga | K-fold estratificado repetido, AUPR y guardrail anti-fuga |
| 6. Modelos y comparación | Candidatos, regularización del boosting, validación cruzada y baseline de partida |
| 7. Ensemble y selección | Decisión final, curva PR, análisis del fraude no detectado e importancia de variables |
| 8. Scoring de producción | Ajuste final, CSV y distribución de scores |
| 9. Traza y comprobación | Tiempos por paso y contraste de las cifras citadas |
| 10. Reflexión final | Resultados, limitaciones, riesgos y mejoras |"""

MD_CARGA = """## 3. Carga y auditoría de datos

Los CSV (o su `.zip`) se buscan en `./`, `./data/` y `../data/`. Si faltan, el notebook **se detiene con un error** que indica el fichero y las opciones: copiarlos a `./data/`, ejecutar con `DESCARGAR_DATOS=true` (descarga desde Drive las mismas copias que usa el notebook de partida, pensado para Colab) o con `DEMO_MODE=true`. Solo este último modo usa **datos sintéticos** con fraude inyectado, para que el notebook sea ejecutable en cualquier entorno; nunca se activa de forma silenciosa.

Qué se comprueba y por qué:
- **Forma, nulos y tipos**: determinan si hace falta imputar o codificar.
- **Duplicados**: hay filas idénticas; si caen en folds distintos, la validación se infla. Se **deduplican** antes de particionar.
- **Desbalanceo**: con ~0,17 % de fraude, el accuracy no sirve y los folds deben ser estratificados.
- **Producción vs. construcción**: si las distribuciones difieren, la validación no generaliza; se mide la diferencia de medias tipificada."""

MD_FEATURES = """## 4. Ingeniería de variables

Las variables `X1..X20` parecen componentes de una proyección (media ~0, casi incorrelacionadas) con **colas muy pesadas** (mínimos de hasta -14 544 con desviaciones de ~150-400). El importe es muy asimétrico (mediana 33, máximo ~29 000).

Decisión: `arcsinh(x/100)` en las variables y `log1p` en el importe. Son **constantes fijas**, no estadísticas aprendidas de los datos, así que no pueden filtrar información de validación ni de producción.

La celda de evidencia (más abajo) mide el efecto en una partición. Los árboles son invariantes a transformaciones monótonas, así que no deberían cambiar; para el GMM la hipótesis era que ayudaría, pero **la medición no lo confirma** (ver resultado y reflexión). Se mantiene por ser inocua y por la robustez numérica de los modelos de densidad, no por una mejora demostrada."""

MD_VALIDACION = """## 5. Validación y control de fuga

**Esquema.** K-fold **estratificado repetido** con semillas fijas. No hay fecha en los datos, así que no es posible una validación temporal. Los folds se construyen *después* de deduplicar.

**Métrica.** AUPR implementado en numpy (equivale a `average_precision_score`, con empates agrupados) y precisión@k (k = nº de fraudes), una métrica complementaria más intuitiva que el AUC-ROC cuando hay un 0,17 % de positivos.

**Guardrail anti-fuga** (`leakage_guard`). Falla con `ErrorFuga` si: (a) se ajusta con filas de validación, (b) la columna objetivo es una variable de entrada, (c) se ajusta sobre producción (no autorizado: hay etiquetas y la deriva es ~0), (d) se calculan estadísticas globales antes del split (la única vía sancionada es `estadisticas_de_ajuste`)."""

MD_MODELOS = """## 6. Modelos y comparación

Candidatos:
| Modelo | Tipo | Idea |
|---|---|---|
| `hgb` | supervisado | Gradient boosting, **muy regularizado** (l2=10, hojas mínimas=50, 8 hojas, lr=0,05) |
| `rf` / `et` | supervisado | Random Forest / Extra Trees, 300 árboles, `class_weight=balanced_subsample` |
| `gmm3` | semisupervisado | Mezcla gaussiana sobre filas normales (como el notebook de partida) |
| `iforest` | semisupervisado | Isolation Forest sobre filas normales |

**Regularizar el boosting es imprescindible.** Con los parámetros por defecto de scikit-learn el AUPR colapsa y es inestable; la celda siguiente lo mide en una partición (HGB por defecto {hgb_defecto} frente a {hgb_reg} regularizado).

El **orquestador** ejecuta los detectores en paralelo (hilos), valida cada salida (longitud, finitud, sin NaN), reintenta una vez con otra semilla y, si un detector *opcional* sigue fallando, entrega el resto con un aviso."""

MD_BASELINE = """### Baseline del notebook de partida, recalculado

El notebook de partida ajustaba un GMM de 5 componentes sobre las variables crudas. Aquí se evalúa con las mismas particiones de la repetición 0, ajustado solo con filas normales del conjunto de ajuste y puntuado fuera de pliegue (score = -log-verosimilitud). El AUPR medio por fold es la referencia {aupr_base_gmm5} citada en el resumen."""

MD_ENSEMBLE = """## 7. Ensemble y selección final

Cada score se convierte a **rangos en [0, 1]** y se promedian. Los rangos hacen comparables escalas muy distintas (probabilidades frente a log-verosimilitudes) y el AUPR solo depende del orden.

**Decisión: `hgb + rf + et`.**
- Validación 5x1 de la tabla de la sección 6: ensemble {aupr_ens} · hgb {aupr_hgb} · hgb+rf {aupr_hgb_rf} · rf {aupr_rf} · et {aupr_et} · ensemble con GMM {aupr_ens_gmm} · gmm3 {aupr_gmm3}.
- El **GMM empeora** el ensemble ({aupr_ens_gmm} < {aupr_ens}): se descarta.
- La ganancia del ensemble sobre `hgb` solo (+{gan_ens_hgb}) es **mucho menor que la desviación entre folds ({desv_min} a {desv_max})**. Se elige para reducir varianza y por robustez ante la distribución de producción, **no porque se haya demostrado que sea mejor**. El coste es mayor tiempo de ajuste, sobre todo en `rf` y `et`: los tiempos dependen de la máquina y se ven en la columna `segundos_total` de la tabla de CV y en la traza de la sección 9."""

MD_CURVA = """### Curva precisión-recall fuera de pliegue

Cada fila de construcción se puntúa una sola vez (repetición 0), con un modelo que no la vio al ajustarse. La figura junta esos scores y muestra la curva de `ens_final`, `hgb`, `gmm3` y el GMM(5) crudo de partida. La línea punteada es la prevalencia (el AUPR de un score al azar).

Nota: el ensemble usa rangos calculados **dentro de cada fold**; al juntar los folds, el AUPR agregado de la leyenda puede diferir algo de la media por fold de la tabla de CV."""

MD_ERRORES = """### Análisis de errores: subgrupos de fraude

Los fraudes se agrupan en dos subpoblaciones con KMeans (k=2) sobre las primeras `N_VARS_AGRUPACION` (20) variables transformadas y tipificadas, usando solo las filas de fraude. El grupo mayor se llama "fuerte" y el menor "suave". Es una agrupación **descriptiva**: usa las etiquetas para describir el resultado y **no entra en ningún modelo**.

Para cada grupo se mide el recall entre los k scores más altos de cada fold (k = nº de fraudes del fold), con los scores fuera de pliegue del ensemble en la repetición 0, y qué parte del fraude no detectado pertenece al grupo suave."""

MD_IMPORTANCIA = """### Importancia de variables (permutación)

Caída media del AUPR de `hgb` en la partición 0 al permutar cada variable de validación (3 repeticiones, semilla fija). Muestra qué variables sostienen el rendimiento; los nombres de las variables son anónimos."""

MD_PRODUCCION = """## 8. Scoring de producción y CSV

Se ajusta con **todo el fichero de construcción deduplicado** y se puntúa producción en su orden original. El CSV tiene **una columna** `score` (cabecera, sin índice), mayor = más probable fraude.

Nota sobre rangos: la media de rangos se calcula sobre el propio vector de scores de producción. No ajusta ningún parámetro con producción; solo ordena y combina las salidas."""

MD_TRAZA = """## 9. Traza de ejecución

Tabla de la última ejecución: paso, modelo, segundos y nota (incluye reintentos y avisos de degradación)."""

MD_COMPROBACION = """### Comprobación de las cifras citadas

Cada cifra citada en el texto se contrasta con el valor calculado en esta ejecución. Si alguna no cuadra, el notebook **avisa y continúa** (las cifras del texto corresponden a la ejecución de referencia con las versiones de `requirements.txt`; otras versiones o sistemas pueden mover los últimos decimales sin que el modelo falle). Con `COMPROBACION_ESTRICTA=true` el notebook se detiene ante cualquier diferencia. En modo demo, o con `SEED` distinto de 42 o `CV_REPETICIONES` distinto de 1, la comprobación se omite (las cifras corresponden a los datos reales con esa configuración)."""

MD_REFLEXION = """## 10. Reflexión final

**Resultados.** El enfoque supervisado con árboles regularizados alcanza ~{aupr_ens} de AUPR en validación sin fuga (5 folds x 1 repetición), más del doble que el GMM del notebook de partida ({aupr_base_gmm5}, recalculado aquí). Las dos palancas que más pesaron fueron *usar las etiquetas* y *regularizar el boosting*; la transformación de variables no mostró una mejora medible (HGB {evid_hgb_crudo} en crudo frente a {evid_hgb_trans} transformado; GMM(3) {evid_gmm_crudo} crudo frente a {evid_gmm_trans} transformado en la partición de prueba), de modo que es una decisión de higiene y no una palanca de rendimiento.

**Limitaciones.**
- Pocos positivos: {n_fraudes_dedup} tras deduplicar (~57 por fold). El AUPR por fold varía con desviaciones de {desv_min} a {desv_max}, de modo que distinguir `hgb`, `rf`, `et` y su ensemble es imposible con esta evidencia.
- Los hiperparámetros se eligieron mirando estos mismos folds (exploración en 3 de ellos), por lo que la cifra de validación es algo optimista. No hay un conjunto de test independiente; una validación anidada lo corregiría.
- **Hay un subgrupo de fraude que el modelo apenas detecta.** El análisis de errores de la sección 7 agrupa los fraudes en dos subpoblaciones: un grupo "fuerte" ({n_fuerte} fraudes), muy desplazado en casi todas las variables, y uno "suave" ({n_suave} fraudes), con desplazamientos leves. Con los scores fuera de pliegue, el ensemble recupera una fracción {recall_fuerte} del grupo fuerte y solo {recall_suave} del suave, y el grupo suave aporta una fracción {frac_suave_no_detectado} del fraude no detectado. Es el límite principal del AUPR y no se resuelve con más complejidad de modelo con estas variables.
- Las variables son anónimas: no se pueden construir variables de negocio (frecuencia por cuenta, recencia, comercio) ni validar en el tiempo.

**Riesgos de generalización.** Producción es muy parecida a construcción (diferencia de medias tipificada < {deriva_max}), lo que favorece que la validación sea representativa; si el fraude en producción cambiara de patrón, un modelo supervisado lo detectaría peor que uno de densidad.

**Qué se haría con más tiempo.** Validación anidada y búsqueda de hiperparámetros con intervalos de confianza; más semillas por modelo; calibrar el peso de cada miembro; estudiar los falsos positivos de mayor score; y, solo si la evidencia lo justifica, probar un modelo con interacciones entre variables."""

C_IMPORTS = '''import os
import socket
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol, Sequence, runtime_checkable
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
)
from sklearn.metrics import precision_recall_curve
from sklearn.mixture import GaussianMixture

pd.options.display.max_columns = None
warnings.filterwarnings("ignore", category=FutureWarning)'''

C_CONFIG = '''DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
# Colab: descarga los CSV de Drive (mismas copias que el notebook de partida) si no están en disco
DESCARGAR_DATOS = os.environ.get("DESCARGAR_DATOS", "false").lower() == "true"
SEED = int(os.environ.get("SEED", "42"))
SALIDA_CSV = os.environ.get("OUTPUT_CSV", "scores.csv")
CV_REPETICIONES = int(os.environ.get("CV_REPETICIONES", "1"))  # 5 particiones por repetición; más repeticiones, más tiempo
N_JOBS = 2
# Comprobación de cifras: por defecto solo avisa (otras versiones de librerías pueden mover los decimales); en modo estricto detiene el notebook
COMPROBACION_ESTRICTA = os.environ.get("COMPROBACION_ESTRICTA", "false").lower() == "true"
TIMEOUT_DESCARGA = 120  # segundos de espera de red por operación

CARPETAS_DATOS = (".", "data", "../data")
NOMBRES_DATOS = {"construccion": "credit_card_construction.csv", "produccion": "credit_card_production.csv"}
URLS_DATOS = {
    "construccion": "https://drive.usercontent.google.com/download?id=1IyihCpU_krumRnPBvMR0ytEfzv_Z2Jg3&export=download&confirm=t&uuid=9699f0e2-e760-49fc-b12e-49f140095280",
    "produccion": "https://drive.usercontent.google.com/download?id=1N5xRbkchTR7FLskuIEmZPYHklsRVJy_q&export=download&confirm=t&uuid=9699f0e2-e760-49fc-b12e-49f140095280",
}


def descargar_zip(url: str, destino: Path, timeout: int = TIMEOUT_DESCARGA) -> Path:
    """Descarga a un .tmp, valida que sea un zip y solo entonces lo renombra a `destino`."""
    tmp = destino.with_name(destino.name + ".tmp")
    socket.setdefaulttimeout(timeout)
    try:
        urlretrieve(url, tmp)
    except (URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Fallo de red al descargar {destino.name}: {e}") from e
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"La descarga de {destino.name} no es un zip válido (¿página de Drive en vez del fichero?).")
    tmp.replace(destino)
    return destino


def localizar(nombre: str) -> Path | None:
    """Busca el CSV (o su .zip, que pandas lee directo) en ./, ./data y ../data."""
    for carpeta in CARPETAS_DATOS:
        for candidato in (nombre, nombre + ".zip"):
            ruta = Path(carpeta) / candidato
            if ruta.exists():
                return ruta
    return None


rutas = {clave: localizar(nombre) for clave, nombre in NOMBRES_DATOS.items()}
if DESCARGAR_DATOS and not DEMO_MODE:
    for clave, ruta in rutas.items():
        if ruta is None:
            destino = Path(NOMBRES_DATOS[clave] + ".zip")
            print(f"Descargando {destino} ...")
            rutas[clave] = descargar_zip(URLS_DATOS[clave], destino)
RUTA_CONSTRUCCION, RUTA_PRODUCCION = rutas["construccion"], rutas["produccion"]

USAR_DEMO = DEMO_MODE  # los datos sintéticos solo se usan si se piden de forma explícita
if not USAR_DEMO:
    faltan = [NOMBRES_DATOS[c] for c, r in rutas.items() if r is None]
    if faltan:
        raise FileNotFoundError(
            f"No se encuentra: {', '.join(faltan)} (carpetas buscadas: {', '.join(CARPETAS_DATOS)}, con o sin .zip). "
            "Opciones: copiarlos a ./data/, ejecutar con DESCARGAR_DATOS=true o con DEMO_MODE=true (datos sintéticos)."
        )
print("MODO DEMO (datos sintéticos)" if USAR_DEMO else f"Datos reales: {RUTA_CONSTRUCCION}, {RUTA_PRODUCCION}")'''


C_CARGA = '''if USAR_DEMO:
    OBJETIVO, COL_IMPORTE = "es_fraude", "importe"
    df_const_bruto = generar_sintetico(6000, seed=SEED)
    df_prod = generar_sintetico(2000, seed=SEED + 1)
    df_prod[OBJETIVO] = np.nan  # como en producción: sin etiqueta
else:
    OBJETIVO, COL_IMPORTE = "Class", "Amount"
    df_const_bruto = pd.read_csv(RUTA_CONSTRUCCION)
    df_prod = pd.read_csv(RUTA_PRODUCCION)

print(f"Construcción: {df_const_bruto.shape} | Producción: {df_prod.shape}")
print("Nulos construcción:", int(df_const_bruto.isna().sum().sum()),
      "| nulos producción (sin la columna objetivo):", int(df_prod.drop(columns=OBJETIVO).isna().sum().sum()))
print("Mismas columnas:", list(df_const_bruto.columns) == list(df_prod.columns))
print(f"Duplicados exactos: construcción {df_const_bruto.duplicated().sum()} | producción {df_prod.duplicated().sum()}")
pos = df_const_bruto[OBJETIVO].sum()
print(f"Fraudes: {int(pos)} de {len(df_const_bruto)} ({100 * pos / len(df_const_bruto):.3f} %)")

df_const = deduplicar(df_const_bruto)
print(f"Tras deduplicar: {len(df_const)} filas, {int(df_const[OBJETIVO].sum())} fraudes")

num = [c for c in df_const.columns if c != OBJETIVO]
dif = ((df_const[num].mean() - df_prod[num].mean()) / df_const[num].std()).abs()
print(f"Deriva producción vs construcción: máx. |dif. de medias| tipificada = {dif.max():.4f} ({dif.idxmax()})")'''

C_FEATURES = '''y = df_const[OBJETIVO].to_numpy(int)
X, COLUMNAS = construir_features(df_const, OBJETIVO, COL_IMPORTE)
X_crudo = df_const[list(COLUMNAS)].to_numpy(float)
print("Variables:", len(COLUMNAS), "| objetivo fuera de las variables:", OBJETIVO not in COLUMNAS)'''

C_EVID_FEATURES = '''cfg_cv = ConfigExperimento(n_splits=5, n_repeats=CV_REPETICIONES, seed=SEED, columna_objetivo=OBJETIVO)
p0 = particiones(y, cfg_cv)[0]
filas = []
for nombre, M in (("crudas", X_crudo), ("transformadas", X)):
    gm = GaussianMixture(3, random_state=SEED).fit(M[p0.idx_fit][y[p0.idx_fit] == 0])
    hgb = GBMSupervisado(SEED).fit(M[p0.idx_fit], y[p0.idx_fit])
    filas.append({"variables": nombre,
                  "AUPR GMM(3)": aupr(y[p0.idx_val], -gm.score_samples(M[p0.idx_val])),
                  "AUPR HGB": aupr(y[p0.idx_val], hgb.score(M[p0.idx_val]))})
evid_features = pd.DataFrame(filas).set_index("variables")
evid_features.round(3)'''

C_REGULARIZACION = '''hgb_defecto = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)
hgb_defecto.fit(X[p0.idx_fit], y[p0.idx_fit])
hgb_reg = GBMSupervisado(SEED).fit(X[p0.idx_fit], y[p0.idx_fit])
evid_reg = pd.DataFrame({
    "modelo": ["HGB por defecto", "HGB regularizado (elegido)"],
    "AUPR": [aupr(y[p0.idx_val], hgb_defecto.predict_proba(X[p0.idx_val])[:, 1]),
             aupr(y[p0.idx_val], hgb_reg.score(X[p0.idx_val]))],
})
evid_reg.round(3)'''

C_CV = '''catalogo_det = catalogo()
especs_cv = list(catalogo_det.values())
combinaciones = {
    "ens_final (hgb+rf+et)": ENSEMBLE_FINAL,
    "ens_con_gmm": (*ENSEMBLE_FINAL, "gmm3"),
    "hgb+rf": ("hgb", "rf"),
}
OOF: dict[str, np.ndarray] = {}  # scores fuera de pliegue de la repetición 0
t0 = time.perf_counter()
res_cv, traza_cv = evaluar_cv(especs_cv, combinaciones, X, y, cfg_cv, COLUMNAS, n_jobs=N_JOBS, oof=OOF)
tabla_cv = (res_cv.groupby("modelo").agg(AUPR=("aupr", "mean"), desv_entre_folds=("aupr", "std"), P_at_k=("p_at_k", "mean"))
            .join(traza_cv.tabla().groupby("modelo").segundos.sum().rename("segundos_total"))
            .sort_values("AUPR", ascending=False).round(4))
print(f"{len(res_cv) // len(tabla_cv)} particiones | {time.perf_counter() - t0:.0f} s")
tabla_cv'''

C_BASELINE = '''folds0 = particiones(y, cfg_cv)[:cfg_cv.n_splits]  # repetición 0: cada fila se valida una vez
oof_base = np.full(len(y), np.nan)
aupr_base = []
for p in folds0:
    verificar_ajuste(p.idx_fit, p)
    normales = p.idx_fit[y[p.idx_fit] == 0]
    gm5 = GaussianMixture(5, random_state=SEED).fit(X_crudo[normales])
    oof_base[p.idx_val] = -gm5.score_samples(X_crudo[p.idx_val])
    aupr_base.append(aupr(y[p.idx_val], oof_base[p.idx_val]))
BASE_GMM5 = float(np.mean(aupr_base))
print(f"GMM(5) crudo (notebook de partida): AUPR medio por fold = {BASE_GMM5:.4f} (desv. {np.std(aupr_base, ddof=1):.4f})")'''

C_CURVA = '''NOMBRE_ENS = "ens_final (hgb+rf+et)"
curvas = {NOMBRE_ENS: OOF[NOMBRE_ENS], "hgb": OOF["hgb"], "gmm3": OOF["gmm3"], "GMM(5) crudo (partida)": oof_base}
fig, ax = plt.subplots(figsize=(7, 5))
for nombre, s in curvas.items():
    prec, rec, _ = precision_recall_curve(y, s)
    ax.plot(rec, prec, label=f"{nombre} (AUPR agregado {aupr(y, s):.3f})")
ax.axhline(y.mean(), ls=":", color="gray", label=f"prevalencia ({y.mean():.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisión")
ax.set_title("Curva precisión-recall fuera de pliegue (repetición 0)")
ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.show()'''

C_ERRORES = '''N_VARS_AGRUPACION = 20  # primeras variables transformadas usadas en el KMeans
SUBGRUPO: dict[str, float] = {}
if int(y.sum()) < 10:
    print("Análisis omitido: menos de 10 fraudes.")
else:
    Z = X[:, :N_VARS_AGRUPACION]
    Zs = (Z - Z.mean(0)) / Z.std(0)
    idx_fraude = np.flatnonzero(y == 1)
    lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(Zs[idx_fraude])
    if (lab == 0).sum() < (lab == 1).sum():
        lab = 1 - lab  # el grupo mayor es "fuerte"
    grupos = {"fuerte": np.zeros(len(y), bool), "suave": np.zeros(len(y), bool)}
    grupos["fuerte"][idx_fraude[lab == 0]] = True
    grupos["suave"][idx_fraude[lab == 1]] = True
    recalls = {g: [] for g in grupos}
    no_detectado = {g: 0 for g in grupos}
    for p in folds0:
        yv, sv = y[p.idx_val], OOF[NOMBRE_ENS][p.idx_val]
        top = np.zeros(len(yv), bool)
        top[np.argsort(-sv, kind="stable")[:int(yv.sum())]] = True
        for g, m in grupos.items():
            recalls[g].append(recall_en_k(yv, sv, m[p.idx_val]))
            no_detectado[g] += int((m[p.idx_val] & ~top).sum())
    total_no_det = sum(no_detectado.values())
    tabla_grupos = pd.DataFrame({
        "fraudes": {g: int(m.sum()) for g, m in grupos.items()},
        "desplazamiento medio |z|": {g: float(np.abs(Zs[m]).mean()) for g, m in grupos.items()},
        "recall@k (media por fold)": {g: float(np.nanmean(r)) for g, r in recalls.items()},
        "no detectados": no_detectado,
    })
    SUBGRUPO = {
        "n_fuerte": int(grupos["fuerte"].sum()), "n_suave": int(grupos["suave"].sum()),
        "recall_fuerte": float(np.nanmean(recalls["fuerte"])), "recall_suave": float(np.nanmean(recalls["suave"])),
        "frac_suave_no_detectado": no_detectado["suave"] / max(total_no_det, 1),
    }
    print(f"Fracción del fraude no detectado que es del grupo suave: {SUBGRUPO['frac_suave_no_detectado']:.2f} ({no_detectado['suave']} de {total_no_det})")
    display(tabla_grupos.round(3))'''

C_IMPORTANCIA = '''hgb_p0 = GBMSupervisado(SEED).fit(X[p0.idx_fit], y[p0.idx_fit])
imp = importancia_permutacion(hgb_p0, X[p0.idx_val], y[p0.idx_val], n_rep=3, seed=SEED)
top10 = np.argsort(imp)[::-1][:10][::-1]
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh([COLUMNAS[j] for j in top10], imp[top10], color="#4C72B0")
ax.set_xlabel("Caída media del AUPR al permutar")
ax.set_title("Importancia por permutación de hgb (10 mayores)")
ax.grid(axis="x", alpha=0.3)
plt.show()'''

C_DISTRIBUCION = '''fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(scores_prod, bins=50, color="#4C72B0")
ax.set_yscale("log")
ax.set_xlabel("Score (media de rangos)")
ax.set_ylabel("Filas (escala log)")
ax.set_title("Distribución del score en producción")
ax.grid(alpha=0.3)
plt.show()'''


def c_comprobacion() -> str:
    """Celda que contrasta CIFRAS con lo calculado; las cifras salen de CIFRAS (misma fuente que el texto)."""
    citadas = {k: (v, t) for k, (v, t, _) in CIFRAS.items()}
    return f'''if USAR_DEMO:
    print("Comprobación omitida en demo (las cifras citadas corresponden a los datos reales).")
elif SEED != 42 or CV_REPETICIONES != 1:
    print(f"Comprobación omitida: las cifras citadas corresponden a SEED=42 y CV_REPETICIONES=1 (actual: SEED={{SEED}}, CV_REPETICIONES={{CV_REPETICIONES}}).")
else:
    CIFRAS_CITADAS = {citadas!r}  # clave: (valor citado, tolerancia)
    ENS = "ens_final (hgb+rf+et)"
    aupr_cv = tabla_cv["AUPR"]
    desv_cv = tabla_cv.loc[["hgb", "rf", "et", ENS, "hgb+rf", "ens_con_gmm"], "desv_entre_folds"]
    calculado = {{
        "n_const": len(df_const_bruto), "n_prod": len(df_prod), "n_fraudes": int(pos),
        "n_dedup": len(df_const), "n_fraudes_dedup": int(df_const[OBJETIVO].sum()),
        "deriva_max": float(dif.max()),
        "aupr_ens": aupr_cv[ENS], "aupr_hgb": aupr_cv["hgb"], "aupr_hgb_rf": aupr_cv["hgb+rf"],
        "aupr_rf": aupr_cv["rf"], "aupr_et": aupr_cv["et"],
        "aupr_ens_gmm": aupr_cv["ens_con_gmm"], "aupr_gmm3": aupr_cv["gmm3"],
        "gan_ens_hgb": aupr_cv[ENS] - aupr_cv["hgb"],
        "desv_min": desv_cv.min(), "desv_max": desv_cv.max(),
        "aupr_base_gmm5": BASE_GMM5,
        "evid_gmm_crudo": evid_features.loc["crudas", "AUPR GMM(3)"], "evid_gmm_trans": evid_features.loc["transformadas", "AUPR GMM(3)"],
        "evid_hgb_crudo": evid_features.loc["crudas", "AUPR HGB"], "evid_hgb_trans": evid_features.loc["transformadas", "AUPR HGB"],
        "hgb_defecto": evid_reg["AUPR"][0], "hgb_reg": evid_reg["AUPR"][1],
        **SUBGRUPO,
    }}
    comprobacion = comparar_cifras(CIFRAS_CITADAS, calculado)
    print(comprobacion.to_string(index=False))
    fallan = comprobacion[~comprobacion["ok"]]
    if fallan.empty:
        print("Todas las cifras citadas coinciden con lo calculado.")
    elif COMPROBACION_ESTRICTA:
        raise AssertionError("Hay cifras citadas que no cuadran:" + chr(10) + fallan.to_string(index=False))
    else:
        print(f"AVISO: {{len(fallan)}} cifra(s) del texto difieren de lo calculado en esta ejecución (normal si las versiones de las librerías o el sistema no son los de requirements.txt). Las que difieren:")
        print(fallan.to_string(index=False))
        print("El modelo y el CSV no se ven afectados. Para detener el notebook ante cualquier diferencia: COMPROBACION_ESTRICTA=true.")'''


C_GUARD = '''ejemplos = {
    "ajustar con filas de validación": lambda: verificar_ajuste(np.arange(90), Particion(np.arange(80), np.arange(80, 100))),
    "objetivo como variable": lambda: verificar_features(list(COLUMNAS) + [OBJETIVO], OBJETIVO),
    "ajustar sobre producción": lambda: verificar_ajuste_no_produccion(X, X),
    "particiones solapadas": lambda: Particion(np.array([1, 2]), np.array([2, 3])),
}
for caso, accion in ejemplos.items():
    try:
        accion()
        print(f"FALLO: '{caso}' no fue detectado")
    except ErrorFuga as e:
        print(f"Bloqueado correctamente -> {caso}: {e}")'''



C_PRODUCCION = '''especs_finales = [catalogo_det[n] for n in ENSEMBLE_FINAL]
scores_prod, resultado = entrenar_y_puntuar(especs_finales, df_const, df_prod, OBJETIVO, COL_IMPORTE, SEED, N_JOBS)
escribir_scores(scores_prod, SALIDA_CSV)
for aviso in resultado.avisos:
    print("AVISO:", aviso)
problemas = validar_csv(SALIDA_CSV, len(df_prod))
assert not problemas, problemas
print(f"{SALIDA_CSV}: {len(scores_prod)} filas, 1 columna, válido. Rango [{scores_prod.min():.3f}, {scores_prod.max():.3f}]")
pd.read_csv(SALIDA_CSV).head()'''

C_TRAZA = "resultado.traza.tabla()"


META_APOYO = {"jupyter": {"source_hidden": True}, "collapsed": False}
ENCABEZADO_APOYO = "Código de apoyo (se puede plegar)"


def apoyo(codigo: str):
    """Celda de infraestructura inlineada: aparece plegada en JupyterLab y VS Code."""
    return new_code_cell(codigo, metadata=META_APOYO)


def construir(miembros: list[str]) -> nbformat.NotebookNode:
    """Devuelve el notebook completo."""
    f = fuente_inline
    md = lambda texto: new_markdown_cell(texto.format(**TXT))  # noqa: E731 - cifras desde CIFRAS
    celdas = [
        new_markdown_cell("# Detección de transacciones bancarias fraudulentas\n\nPráctica de detección de anomalías · Máster MIAX"),
        new_markdown_cell("## 1. Integrantes del grupo"),
        new_code_cell("MIEMBROS = " + repr(miembros) + "\nfor m in MIEMBROS:\n    print('-', m)"),
        md(MD_RESUMEN),
        md(MD_MAPA),
        new_markdown_cell("## Entorno y configuración\n\nSolo se usan las librerías de `requirements.txt` (numpy, pandas, scikit-learn, pydantic, joblib, matplotlib para las figuras). Todo el código está en este notebook."),
        new_code_cell(C_IMPORTS),
        new_code_cell(C_CONFIG),
        new_markdown_cell(f"### {ENCABEZADO_APOYO}: datos sintéticos de reserva (modo demo)"),
        apoyo(f(["samples/generar_sintetico.py"])),
        new_markdown_cell(f"### {ENCABEZADO_APOYO}: utilidades de validación (deduplicación, AUPR, particiones, diagnósticos)\n\nSe definen aquí porque la auditoría ya necesita deduplicar. Dependen del contrato `Detector` y del guardrail anti-fuga, incluidos antes."),
        apoyo(f(["src/anomalias/detectors/base.py", "src/anomalias/domain/leakage_guard.py", "src/anomalias/domain/schema.py"])),
        apoyo(f(["src/anomalias/domain/ensemble.py", "src/anomalias/domain/validation.py", "src/anomalias/domain/diagnostico.py"])),
        md(MD_CARGA),
        new_code_cell(C_CARGA),
        md(MD_FEATURES),
        new_markdown_cell(f"### {ENCABEZADO_APOYO}: transformación de variables"),
        apoyo(f(["src/anomalias/domain/features.py"])),
        new_code_cell(C_FEATURES),
        new_markdown_cell(f"### {ENCABEZADO_APOYO}: detectores"),
        apoyo(f(["src/anomalias/detectors/gbm_supervisado.py", "src/anomalias/detectors/bosques.py",
                 "src/anomalias/detectors/gmm.py", "src/anomalias/detectors/isolation_forest.py",
                 "src/anomalias/detectors/registro.py"])),
        new_markdown_cell("### Evidencia: efecto de la transformación de variables (una partición)"),
        new_code_cell(C_EVID_FEATURES),
        md(MD_VALIDACION),
        new_code_cell(C_GUARD),
        md(MD_MODELOS),
        new_markdown_cell(f"### {ENCABEZADO_APOYO}: orquestación y generación del CSV"),
        apoyo(f(["src/anomalias/orchestration/trace.py", "src/anomalias/orchestration/pipeline.py",
                 "src/anomalias/orchestration/submission.py"])),
        new_markdown_cell("### Evidencia: regularización del boosting (misma partición)"),
        new_code_cell(C_REGULARIZACION),
        new_markdown_cell("### Validación cruzada de todos los candidatos y ensembles\n\n`CV_REPETICIONES=1` (5 particiones) para acotar el tiempo. La tabla resultante es la fuente de las cifras que se citan en las secciones 2, 7 y 10."),
        new_code_cell(C_CV),
        md(MD_BASELINE),
        new_code_cell(C_BASELINE),
        md(MD_ENSEMBLE),
        md(MD_CURVA),
        new_code_cell(C_CURVA),
        md(MD_ERRORES),
        new_code_cell(C_ERRORES),
        md(MD_IMPORTANCIA),
        new_code_cell(C_IMPORTANCIA),
        md(MD_PRODUCCION),
        new_code_cell(C_PRODUCCION),
        new_code_cell(C_DISTRIBUCION),
        md(MD_TRAZA),
        new_code_cell(C_TRAZA),
        md(MD_COMPROBACION),
        new_code_cell(c_comprobacion()),
        md(MD_REFLEXION),
    ]
    nb = new_notebook(cells=celdas)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miembros", help='Integrantes separados por ";" (por defecto, MIEMBROS de .env)')
    ap.add_argument("--salida", default=str(RAIZ / "entrega" / "taller_anomalias.ipynb"))
    ap.add_argument("--permitir-pendiente", action="store_true", help="Borrador: permite integrantes sin definir")
    a = ap.parse_args()
    miembros = [m.strip() for m in a.miembros.split(";") if m.strip()] if a.miembros else list(cargar_config().miembros)
    try:
        validar_miembros(miembros)
    except ValueError as e:
        if not a.permitir_pendiente:
            sys.exit(f"ERROR: {e}")
        miembros = ["PENDIENTE: completar integrantes antes de entregar"]
        print("AVISO: notebook BORRADOR sin integrantes; no es entregable.")
    nb = construir(miembros)
    nbformat.validate(nb)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, a.salida)
    print("Notebook escrito en", a.salida)


if __name__ == "__main__":
    main()
