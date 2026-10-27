# 02 · Especificación técnica — Detección de anomalías en transacciones

Complementa `01_MATRIZ_REQUISITOS.md`. Si algo choca, manda la matriz; si es ambiguo, **preguntar**.

## 1. Alcance
Sistema que asigna un **score de anomalía/fraude** a cada transacción del fichero de producción. Se desarrolla de forma modular y se **empaqueta en un único notebook autocontenido** + un CSV de una columna.

## 2. Arquitectura

```
anomalias/
├── app: no hay. Entregables en entrega/
├── data/                       # datos reales (gitignored)
├── docs/                       # 01 matriz, 02 spec, 03 decisiones (bitácora)
├── src/anomalias/
│   ├── config.py               # rutas, semilla, DEMO_MODE, nombres de columnas (por env/constantes)
│   ├── detectors/
│   │   ├── base.py             # Protocol Detector: fit(X, y=None), score(X) -> ndarray
│   │   ├── isolation_forest.py # ÚNICOS módulos que importan sklearn/otras libs de modelado
│   │   ├── lof.py
│   │   ├── gbm_supervisado.py  # solo si hay etiquetas (P-02)
│   │   └── mock.py             # detector trivial para tests
│   ├── domain/
│   │   ├── schema.py           # Pydantic: contrato de datos, ConfigExperimento, ResultadoDetector
│   │   ├── features.py         # ingeniería de variables (pura, sin I/O)
│   │   ├── validation.py       # esquema de validación + métricas (según P-01)
│   │   ├── ensemble.py         # normalización por rangos y combinación de scores
│   │   └── leakage_guard.py    # comprobaciones deterministas anti-fuga
│   └── orchestration/
│       ├── pipeline.py         # carga -> features -> detectores (en paralelo) -> ensemble -> scores
│       └── trace.py            # traza: (paso, modelo, segundos, nota)
├── scripts/
│   ├── build_notebook.py       # ensambla notebook autocontenido desde src/ + celdas narrativas
│   └── validate_submission.py  # valida el CSV (R-06)
├── samples/
│   └── generar_sintetico.py    # transacciones sintéticas con fraude inyectado (modo demo/tests)
├── tests/                      # solo datos sintéticos; cero dependencia de datos reales
└── entrega/
    ├── taller_anomalias.ipynb
    └── scores.csv
```

## 3. Reglas de arquitectura (obligatorias)

1. **Estructura** como arriba. Notebook de partida del Drive se conserva intacto en `notebooks/`.
2. `domain/` y `orchestration/` dependen solo del Protocol `Detector` (en `detectors/base.py`). Ninguna librería de modelado (sklearn, lightgbm, torch…) se importa fuera de `detectors/`.
3. **Modo demo**: si `DEMO_MODE=true` o faltan los datos reales, el pipeline corre completo con datos sintéticos. Los tests usan solo sintéticos.
   - **Excepción (PR 1, aceptada por Piettro):** el **notebook** ya no cae a demo si faltan los datos: se detiene con un error claro y solo usa datos sintéticos con `DEMO_MODE=true` (admite también `.zip` y `DESCARGAR_DATOS=true`). `scripts/generar_scores.py` mantiene la caída a demo, pero escribe `scores_demo.csv`, nunca `scores.csv`.
4. Toda salida de un detector se **valida con Pydantic/numpy checks**: longitud = nº filas, finita, sin NaN. Si falla, reintentar una vez (p. ej. con otra semilla) y degradar con mensaje claro.
5. El orquestador ejecuta detectores **en paralelo** (joblib) y registra la traza por paso, mostrada en el notebook.
6. **Degradación elegante**: si un detector opcional falla, el pipeline entrega igualmente el score del resto y avisa.
7. **Guardrail anti-fuga (determinista)**: tests y asserts que impiden (a) ajustar con filas de validación, (b) usar la columna objetivo como feature, (c) ajustar sobre producción salvo decisión P-05, (d) estadísticas globales calculadas antes del split.
8. **Cero datos y cero secretos en el repo**: `data/`, `.env` en `.gitignore`; `.env.example` con rutas y semilla.
9. Código limpio: type hints, funciones pequeñas, docstrings breves en español, sin dependencias innecesarias.
10. **Español** en README, docs, comentarios, textos del notebook.
11. **Notebook autocontenido**: `build_notebook.py` inlinea el código de `src/` en celdas (sin `import anomalias`). Debe ejecutar de arriba abajo en un entorno limpio con solo `requirements.txt`.

## 4. Estrategia de modelado (a confirmar con los datos)

Árbol de decisión, resuelto en la Fase 0 con evidencia:

- **Con etiquetas (P-02)**: baseline supervisado (gradient boosting) + detectores no supervisados como features/ensemble; validación **temporal** si hay fecha, estratificada si no; métrica de P-01.
- **Sin etiquetas**: ensemble no supervisado (Isolation Forest, LOF, estimadores robustos, distancia de Mahalanobis robusta, reconstrucción PCA) con **promedio de rangos**; validación con fuga cero y, si es posible, un conjunto sintético/etiquetado parcial para sanity-check.
- Ingeniería de variables: transformaciones de importe (log), frecuencia y recencia por cuenta/comercio **calculadas sin mirar el futuro**, codificación de categóricas de alta cardinalidad (conteo/target encoding solo dentro de folds).
- Convención de score: **mayor = más anómalo** (confirmar P-03). Salida normalizada a [0, 1] por rangos.

## 5. Contrato del CSV de producción
- Una columna, `len == nº filas del fichero de producción`, mismo orden, floats finitos.
- Cabecera según P-04. Validado por `scripts/validate_submission.py` antes de dar nada por cerrado.

## 6. Estructura del notebook final (R-01…R-05)
1. Título + **integrantes** (R-02)
2. Resumen ejecutivo del enfoque y resultado de validación
3. Carga y auditoría de datos
4. Ingeniería de variables (con justificación)
5. Validación: esquema, métrica, anti-fuga (con justificación)
6. Modelos y comparación (tabla con métricas y tiempos)
7. Ensemble y selección final (con justificación)
8. Scoring de producción y escritura del CSV + validación del CSV
9. Traza de ejecución
10. **Reflexión final** (R-05): resultados, limitaciones, riesgos, mejoras

## 7. Plan por fases (cada una termina con tests + informe + confirmación)

| Fase | Contenido | Criterio de salida |
|---|---|---|
| 0 | Esqueleto del repo, entorno, sintético, tests de humo, **auditoría de datos y resolución de P-01…P-08** | Tests verdes; informe de auditoría; respuestas a preguntas abiertas |
| 1 | Marco de validación + guardrail anti-fuga + baselines | Métrica baseline reproducible sin fuga |
| 2 | Detectores, ingeniería de variables, ensemble, traza | Mejora medida sobre baseline, con tiempos |
| 3 | Scoring de producción, CSV, `build_notebook.py`, `validate_submission.py` | CSV válido; notebook generado |
| 4 | Verificación final: ejecución limpia del notebook, revisión contra la matriz, congelación | Todos los R-xx en "Cumplido" antes del **28/10** (colchón de un día) |

## 8. Ampliación: EDA de referencia (Fases 5 y 6)

El notebook `docs/EDA_datos.ipynb` (generado por `scripts/build_eda.py`; dependencias en `requirements-eda.txt`) es un **documento de apoyo, no un entregable**: explica con evidencia cada decisión del proyecto. No sustituye a los dos entregables de `entrega/`.

| Fase | Contenido | Criterio de salida |
|---|---|---|
| 5 | **Mejorar el EDA**: revisar cada afirmación contra las cifras, profundizar en los hallazgos abiertos (importes exactos, número de variables extremas, dependencia no lineal entre X, subgrupos de fraude), pulir narrativa y figuras | Cada frase con cifra verificada por la ejecución; hipótesis H1…H8 bien formuladas; EDA ejecuta de arriba abajo sin errores |
| 6 (hecha) | **Validar el proyecto contra el EDA**: contrastar H1…H8 con el pipeline real (validación sin fuga) y decidir si alguna conclusión del EDA obliga a cambiar el proyecto | Tabla «hipótesis → resultado → ¿cambia el proyecto?»; si algo cambia, se rehace CSV y notebook y se verifica de nuevo |

Regla: la Fase 6 **no modifica** el pipeline salvo que la evidencia lo justifique y el usuario lo confirme; cualquier mejora se mide con el mismo esquema de validación (5×2 sin fuga).
