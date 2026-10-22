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

**Problema.** Asignar un score de fraude a cada fila de `credit_card_production.csv` (113 923 transacciones). La métrica es el **AUPR** (`average_precision_score`) y mayor score = más probable fraude, según el notebook de partida.

**Datos.** El fichero de construcción tiene 170 884 filas con etiqueta (295 fraudes, 0,17 %). Producción tiene las mismas 21 variables numéricas, sin etiqueta, y una distribución prácticamente idéntica a la de construcción (diferencia de medias tipificada por variable < 0,02; ver la auditoría de la sección 3).

**Enfoque.** Aprendizaje **supervisado** con tres modelos de árboles (gradient boosting regularizado, Random Forest y Extra Trees) combinados por **media de rangos**. Los modelos no supervisados (mezcla gaussiana, Isolation Forest) se evaluaron y se descartaron porque rinden menos y no mejoran el ensemble.

**Resultado de validación** (5 folds x 2 repeticiones, sin fuga, ejecución de referencia): ensemble **AUPR = 0,834**, frente a 0,436 del modelo del notebook de partida (GMM sobre variables crudas). La desviación entre folds es ~0,05 (hay ~57 fraudes por fold), por lo que las diferencias de unas milésimas entre los modelos supervisados **no son significativas**.

**Cómo leer este notebook.** Cada decisión (variables, validación, hiperparámetros, ensemble) va con una celda que la justifica con cifras calculadas en la propia ejecución."""

MD_CARGA = """## 3. Carga y auditoría de datos

Los CSV se buscan en `./`, `./data/` y `../data/`. Si no están (o `DEMO_MODE=true`), el notebook corre completo con **datos sintéticos** con fraude inyectado, para que sea ejecutable en cualquier entorno.

Qué comprobamos y por qué:
- **Forma, nulos y tipos**: determinan si hace falta imputar o codificar.
- **Duplicados**: hay filas idénticas; si caen en folds distintos, la validación se infla. Se **deduplican** antes de particionar.
- **Desbalanceo**: con ~0,17 % de fraude, el accuracy no sirve y los folds deben ser estratificados.
- **Producción vs. construcción**: si las distribuciones difieren, la validación no generaliza; medimos la diferencia de medias tipificada."""

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

**Regularizar el boosting es imprescindible.** Con los parámetros por defecto de scikit-learn el AUPR colapsa y es inestable; lo comprobamos a continuación.

El **orquestador** ejecuta los detectores en paralelo (hilos), valida cada salida (longitud, finitud, sin NaN), reintenta una vez con otra semilla y, si un detector *opcional* sigue fallando, entrega el resto con un aviso."""

MD_ENSEMBLE = """## 7. Ensemble y selección final

Cada score se convierte a **rangos en [0, 1]** y se promedian. Los rangos hacen comparables escalas muy distintas (probabilidades frente a log-verosimilitudes) y el AUPR solo depende del orden.

**Decisión: `hgb + rf + et`.**
- Referencia 5x2: ensemble 0,834 · hgb 0,831 · hgb+rf 0,832 · rf 0,822 · et 0,819 · ensemble con GMM 0,805.
- El **GMM empeora** el ensemble (0,805 < 0,834): se descarta.
- La ganancia del ensemble sobre `hgb` solo (+0,003) es **menor que la desviación entre folds (~0,05)**. Lo elegimos para reducir varianza y por robustez ante la distribución de producción, **no porque se haya demostrado que sea mejor**. El coste es mayor tiempo de ajuste: sobre todo el fichero de construcción, `rf` tarda ~1-1,5 min y `et` ~0,5-1 min (la traza de la sección 9 muestra los tiempos de esta ejecución)."""

MD_PRODUCCION = """## 8. Scoring de producción y CSV

Se ajusta con **todo el fichero de construcción deduplicado** y se puntúa producción en su orden original. El CSV tiene **una columna** `score` (cabecera, sin índice), mayor = más probable fraude.

Nota sobre rangos: la media de rangos se calcula sobre el propio vector de scores de producción. No ajusta ningún parámetro con producción; solo ordena y combina las salidas."""

MD_TRAZA = """## 9. Traza de ejecución

Tabla de la última ejecución: paso, modelo, segundos y nota (incluye reintentos y avisos de degradación)."""

MD_REFLEXION = """## 10. Reflexión final

**Resultados.** El enfoque supervisado con árboles regularizados alcanza ~0,83 de AUPR en validación sin fuga, casi el doble que el GMM del notebook de partida (0,44). Las dos palancas que más pesaron fueron *usar las etiquetas* y *regularizar el boosting*; la transformación de variables no mostró una mejora medible (HGB idéntico; GMM(3) 0,750 crudo frente a 0,732 transformado en la partición de prueba), de modo que es una decisión de higiene y no una palanca de rendimiento.

**Limitaciones.**
- Pocos positivos: 285 tras deduplicar (~57 por fold). El AUPR por fold varía ±0,05, de modo que distinguir `hgb`, `rf`, `et` y su ensemble es imposible con esta evidencia.
- Los hiperparámetros se eligieron mirando estos mismos folds (exploración en 3 de ellos), por lo que la cifra de validación es algo optimista. No tenemos un conjunto de test independiente; una validación anidada lo corregiría.
- **Hay un subgrupo de fraude que el modelo apenas detecta.** Un análisis complementario posterior (EDA y validación adicionales, *fuera de esta entrega* y no recalculado aquí) agrupa los fraudes en dos subpoblaciones estables: un grupo "fuerte" (~3/4), muy desplazado en casi todas las variables, y uno "suave" (~1/4), con desplazamientos leves. Con validación cruzada, el modelo recupera ~99 % del grupo fuerte pero solo ~30 % del suave, que explica la práctica totalidad del fraude no detectado: es el límite principal del AUPR ≈ 0,83 y no se resuelve con más complejidad de modelo con estas variables.
- Las variables son anónimas: no podemos construir variables de negocio (frecuencia por cuenta, recencia, comercio) ni validar en el tiempo.

**Riesgos de generalización.** Producción es muy parecida a construcción (diferencia de medias tipificada < 0,02), lo que favorece que la validación sea representativa; si el fraude en producción cambiara de patrón, un modelo supervisado lo detectaría peor que uno de densidad.

**Qué haríamos con más tiempo.** Validación anidada y búsqueda de hiperparámetros con intervalos de confianza; más semillas por modelo; calibrar el peso de cada miembro; estudiar los falsos positivos de mayor score; y, solo si la evidencia lo justifica, probar un modelo con interacciones entre variables."""

C_IMPORTS = '''import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
)
from sklearn.mixture import GaussianMixture

pd.options.display.max_columns = None
warnings.filterwarnings("ignore", category=FutureWarning)'''

C_CONFIG = '''DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
SEED = int(os.environ.get("SEED", "42"))
SALIDA_CSV = os.environ.get("OUTPUT_CSV", "scores.csv")
CV_REPETICIONES = int(os.environ.get("CV_REPETICIONES", "1"))  # 1 = ~5 min; la ref. 5x2 tarda ~10 min
N_JOBS = 2


def localizar(nombre: str):
    """Busca el CSV en ./, ./data y ../data."""
    for carpeta in (".", "data", "../data"):
        ruta = Path(carpeta) / nombre
        if ruta.exists():
            return ruta
    return None


RUTA_CONSTRUCCION = localizar("credit_card_construction.csv")
RUTA_PRODUCCION = localizar("credit_card_production.csv")
USAR_DEMO = DEMO_MODE or RUTA_CONSTRUCCION is None or RUTA_PRODUCCION is None
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
pd.DataFrame(filas).round(3)'''

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

C_REGULARIZACION = '''hgb_defecto = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)
hgb_defecto.fit(X[p0.idx_fit], y[p0.idx_fit])
hgb_reg = GBMSupervisado(SEED).fit(X[p0.idx_fit], y[p0.idx_fit])
pd.DataFrame({
    "modelo": ["HGB por defecto", "HGB regularizado (elegido)"],
    "AUPR": [aupr(y[p0.idx_val], hgb_defecto.predict_proba(X[p0.idx_val])[:, 1]),
             aupr(y[p0.idx_val], hgb_reg.score(X[p0.idx_val]))],
}).round(3)'''

C_CV = '''catalogo_det = catalogo()
especs_cv = list(catalogo_det.values())
combinaciones = {
    "ens_final (hgb+rf+et)": ENSEMBLE_FINAL,
    "ens_con_gmm": (*ENSEMBLE_FINAL, "gmm3"),
    "hgb+rf": ("hgb", "rf"),
}
t0 = time.perf_counter()
res_cv, traza_cv = evaluar_cv(especs_cv, combinaciones, X, y, cfg_cv, COLUMNAS, n_jobs=N_JOBS)
tabla_cv = (res_cv.groupby("modelo").agg(AUPR=("aupr", "mean"), desv_entre_folds=("aupr", "std"), P_at_k=("p_at_k", "mean"))
            .join(traza_cv.tabla().groupby("modelo").segundos.sum().rename("segundos_total"))
            .sort_values("AUPR", ascending=False).round(4))
print(f"{len(res_cv) // len(tabla_cv)} particiones | {time.perf_counter() - t0:.0f} s")
tabla_cv'''

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


def construir(miembros: list[str]) -> nbformat.NotebookNode:
    """Devuelve el notebook completo."""
    f = fuente_inline
    celdas = [
        new_markdown_cell("# Detección de transacciones bancarias fraudulentas\n\nPráctica de detección de anomalías · Máster MIAX"),
        new_markdown_cell("## 1. Integrantes del grupo"),
        new_code_cell("MIEMBROS = " + repr(miembros) + "\nfor m in MIEMBROS:\n    print('-', m)"),
        new_markdown_cell(MD_RESUMEN),
        new_markdown_cell("## Entorno y configuración\n\nSolo se usan las librerías de `requirements.txt` (numpy, pandas, scikit-learn, pydantic, joblib). Todo el código está en este notebook."),
        new_code_cell(C_IMPORTS),
        new_code_cell(C_CONFIG),
        new_markdown_cell("### Datos sintéticos de reserva (modo demo)"),
        new_code_cell(f(["samples/generar_sintetico.py"])),
        new_markdown_cell("### Utilidades de validación (deduplicación, AUPR, particiones)\n\nSe definen aquí porque la auditoría ya necesita deduplicar. Dependen del contrato `Detector` y del guardrail anti-fuga, incluidos antes."),
        new_code_cell(f(["src/anomalias/detectors/base.py", "src/anomalias/domain/leakage_guard.py", "src/anomalias/domain/schema.py"])),
        new_code_cell(f(["src/anomalias/domain/ensemble.py", "src/anomalias/domain/validation.py"])),
        new_markdown_cell(MD_CARGA),
        new_code_cell(C_CARGA),
        new_markdown_cell(MD_FEATURES),
        new_code_cell(f(["src/anomalias/domain/features.py"])),
        new_code_cell(C_FEATURES),
        new_markdown_cell("### Detectores (código)"),
        new_code_cell(f(["src/anomalias/detectors/gbm_supervisado.py", "src/anomalias/detectors/bosques.py",
                         "src/anomalias/detectors/gmm.py", "src/anomalias/detectors/isolation_forest.py",
                         "src/anomalias/detectors/registro.py"])),
        new_markdown_cell("### Evidencia: efecto de la transformación de variables (una partición)"),
        new_code_cell(C_EVID_FEATURES),
        new_markdown_cell(MD_VALIDACION),
        new_code_cell(C_GUARD),
        new_markdown_cell(MD_MODELOS),
        new_code_cell(f(["src/anomalias/orchestration/trace.py", "src/anomalias/orchestration/pipeline.py",
                         "src/anomalias/orchestration/submission.py"])),
        new_markdown_cell("### Evidencia: regularización del boosting (misma partición)"),
        new_code_cell(C_REGULARIZACION),
        new_markdown_cell("### Validación cruzada de todos los candidatos y ensembles\n\n`CV_REPETICIONES=1` (5 particiones) para acotar el tiempo; la tabla de referencia con 5x2 está en la sección 7."),
        new_code_cell(C_CV),
        new_markdown_cell(MD_ENSEMBLE),
        new_markdown_cell(MD_PRODUCCION),
        new_code_cell(C_PRODUCCION),
        new_markdown_cell(MD_TRAZA),
        new_code_cell(C_TRAZA),
        new_markdown_cell(MD_REFLEXION),
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
