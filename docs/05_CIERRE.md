# 05 · Cierre y congelación (Fase 4, versión 2)

Versión 2 (27/10/2026): el notebook de entrega se sustituye por la propuesta revisada de Raúl (PR 1, con los ajustes de la rama `revision/ajustes-pr1`; ver `docs/06_PROPUESTA_REVISION.md`). Entrega: jueves 29/10/2026, 18:30.

## Entregables congelados
| Fichero | SHA-256 |
|---|---|
| `entrega/scores.csv` | `8b5371a1a93ac4bd9c87c7c46b00b458c9800786d37ea43353bf1229d1cdd257` |
| `entrega/taller_anomalias.ipynb` | `248b5ffeaacbb7739e8ea227aeb0405d3e8240c92984f9f260c6c59907d7f051` |

`entrega/` contiene **solo** esos dos ficheros. Si se vuelve a generar cualquiera de ellos, hay que actualizar esta tabla.

Historial: el `scores.csv` **no ha cambiado** (el modelo es el mismo). El notebook de la versión 1 tenía SHA-256 `0b4232d0ebf8a27a28a7c20d24cb0f5c08bb6d1b6f5f05abf0dec50362c1befb`. `propuesta/taller_anomalias.ipynb` es una copia idéntica del actual.

## Verificación en entorno limpio (notebook v2)
- Entorno nuevo (`py -3.12 -m venv`) con **solo** `requirements.txt`, sin el paquete `anomalias` ni el `.venv` del proyecto; notebook y datos copiados a una carpeta aislada.
- El notebook ejecutó **de arriba abajo sin errores** (49 celdas, 0 sin ejecutar, 3 figuras) en **575 s**, y la celda de comprobación indicó «Todas las cifras citadas coinciden con lo calculado».
- El `scores.csv` generado es **idéntico bit a bit** (mismo SHA-256) al de `entrega/` y al producido por `scripts/generar_scores.py`.
- Versiones: numpy 2.5.4 · pandas 3.1.0 · scikit-learn 1.9.1 · scipy 1.18.1 · pydantic 2.14.1 · joblib 1.6.0 · matplotlib 3.11.2 · Python 3.12.
- Imports del notebook: `numpy, pandas, sklearn, pydantic, joblib, matplotlib` y la biblioteca estándar (`socket`, `urllib` y `zipfile` solo se usan en la descarga opcional de Drive).

## Qué cambió respecto a la versión 1 del notebook
- **Datos**: si faltan los CSV, el notebook **se detiene con un error claro** (antes caía en silencio a datos sintéticos y podía escribir un `scores.csv` de 2 000 filas). Lee también `.zip` y admite `DESCARGAR_DATOS=true` (descarga de Drive; opt-in). El demo solo se activa con `DEMO_MODE=true`.
- **Baseline recalculado dentro del notebook**: GMM(5) crudo del notebook de partida con las mismas particiones: **AUPR 0,375** (5×1). La cifra 0,436 de la Fase 1 (`docs/03`) sigue siendo correcta para su configuración (5×2); no son comparables entre sí.
- **Cifras que cuadran**: se citan solo las que el notebook calcula (5×1) y una celda contrasta 28 cifras con lo calculado. Por defecto **avisa y continúa** si alguna difiere (otras versiones de librerías pueden mover los decimales); `COMPROBACION_ESTRICTA=true` detiene el notebook.
- **Gráficos**: curva precisión-recall fuera de pliegue, distribución del score de producción e importancia por permutación de `hgb`.
- **Subgrupos de fraude calculados dentro del notebook**: recall@k 0,996 (218 fraudes «fuertes») frente a 0,310 (67 «suaves»); el 98 % del fraude no detectado es «suave».
- **Redacción impersonal**, mapa del notebook y celdas de código de apoyo plegadas.

## Revisión contra la matriz
| ID | Estado | Evidencia |
|---|---|---|
| R-01 | Cumplido | Existe `entrega/taller_anomalias.ipynb` (único `.ipynb`) |
| R-02 | Cumplido | Celda 1: Piettro Rodrigues, Alonso Díaz, Raúl Rodríguez; test que rechaza vacío/marcadores |
| R-03 | Cumplido | Sin `import anomalias`; ejecuta en entorno limpio |
| R-04 | Cumplido | Cada decisión con celda de justificación y cifras calculadas en la ejecución (transformación, regularización, validación, CV, ensemble, baseline recalculado) y comprobación de las cifras citadas |
| R-05 | Cumplido | Sección 10: resultados, limitaciones (subgrupo «suave» calculado en el propio notebook), riesgos, mejoras |
| R-06 | Cumplido | `scripts/validate_submission.py`: 1 columna (`score`), 113 923 filas, sin NaN/inf, orden de producción |
| R-07 | Cumplido (estimación) | AUPR de validación ≈ 0,83 (0,834 en 5×1 sin fuga; 0,834 también en 5×2). **Optimista**: hiperparámetros elegidos mirando los mismos pliegues |
| R-08 | Cumplido con una desviación | `requirements.txt` fijado (ahora incluye `matplotlib==3.11.2`). Objetivo Python 3.11; **verificado en 3.12** (3.11 no instalado) |
| D-01 | Cumplido | Mismo hash en ≥ 5 ejecuciones (scripts, notebook v1, notebook v2 en `.venv` y en entorno limpio) |
| D-02 | Cumplido | Guardrail + tests; ajuste solo con construcción; `Class` de producción no influye (test) |
| D-03 | Cumplido | 575 s (9,6 min) en entorno limpio; 548 s en otra ejecución de la propuesta; ~5 min según el autor de la PR en su máquina. Sin límite de tiempo conocido (P-07) |
| D-04 | Cumplido | Traza (paso, modelo, segundos, nota) en la sección 9 |
| D-05 | Cumplido | Test con detector opcional que falla y con uno obligatorio |
| D-06 | Cumplido | `data/` y `.env` en `.gitignore`; `propuesta/scores.csv` no se versiona; sin secretos |
| D-07 | Cumplido | Textos, comentarios y docs en español |

Tests: **54 passed** (rama `revision/ajustes-pr1`).

## Riesgos abiertos (para conocimiento del grupo)
1. **Cabecera del CSV (P-04):** el CSV lleva la cabecera `score`. Si el evaluador lo leyese sin cabecera, la primera fila sería un texto. El enunciado no lo especifica; la decisión se confirmó con el usuario.
2. **Estimación optimista** del AUPR (ver R-07) y **sin conjunto de test independiente**.
3. **Ruido de la métrica ≈ 0,055–0,069 entre pliegues:** `hgb`, `rf`, `et` y su ensemble no son distinguibles; el ensemble se eligió por robustez.
4. **Sin los CSV, el notebook se detiene** (por diseño). Si el evaluador lo reejecutase sin datos obtendría un error claro, no un CSV falso; el `scores.csv` entregado ya está generado. La descarga desde Drive (`DESCARGAR_DATOS=true`) **no se ha probado con la red real**.
5. **Otras versiones de librerías o sistemas** pueden mover los decimales; la comprobación de cifras avisa pero no detiene el notebook por defecto.
6. **Tiempo de ejecución del notebook:** ~5–10 min según la máquina (hasta 15 min en una ejecución anterior con la máquina muy cargada).

## Cómo regenerar (si hiciese falta)
```bash
python scripts/generar_scores.py          # entrega/scores.csv (~2-4 min)
python scripts/validate_submission.py     # valida el CSV
python scripts/build_notebook.py          # lee MIEMBROS de .env; genera el notebook sin ejecutar
cd entrega && python -m nbconvert --to notebook --execute taller_anomalias.ipynb --inplace
```
