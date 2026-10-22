# 05 · Cierre y congelación (Fase 4)

Fecha de la verificación: 22/10/2026 (entrega: jueves 29/10/2026, 18:30; colchón de una semana).

## Entregables congelados
| Fichero | SHA-256 |
|---|---|
| `entrega/scores.csv` | `8b5371a1a93ac4bd9c87c7c46b00b458c9800786d37ea43353bf1229d1cdd257` |
| `entrega/taller_anomalias.ipynb` | `0b4232d0ebf8a27a28a7c20d24cb0f5c08bb6d1b6f5f05abf0dec50362c1befb` |

`entrega/` contiene **solo** esos dos ficheros. Si se vuelve a generar cualquiera de ellos, hay que actualizar esta tabla.

## Verificación en entorno limpio
- Entorno nuevo (`py -3.12 -m venv`) con **solo** `requirements.txt`, sin el paquete `anomalias` ni el `.venv` del proyecto; notebook y datos copiados a una carpeta aislada.
- El notebook ejecutó **de arriba abajo sin errores** (35 celdas, 0 sin ejecutar).
- El `scores.csv` generado es **idéntico bit a bit** (mismo SHA-256) al producido por `scripts/generar_scores.py` y por las ejecuciones anteriores del notebook: reproducibilidad entre entornos y entre scripts.
- Versiones: numpy 2.5.4 · pandas 3.1.0 · scikit-learn 1.9.1 · scipy 1.18.1 · pydantic 2.14.1 · joblib 1.6.0 · Python 3.12.
- Imports del notebook: solo `numpy, pandas, sklearn, pydantic, joblib` y la biblioteca estándar.

## Revisión contra la matriz
| ID | Estado | Evidencia |
|---|---|---|
| R-01 | Cumplido | Existe `entrega/taller_anomalias.ipynb` (único `.ipynb`) |
| R-02 | Cumplido | Celda 1: Piettro Rodrigues, Alonso Díaz, Raúl Rodríguez; test que rechaza vacío/marcadores |
| R-03 | Cumplido | Sin `import anomalias`; ejecuta en entorno limpio |
| R-04 | Cumplido | Cada decisión con celda de justificación y cifras calculadas en la ejecución (transformación, regularización, validación, CV, ensemble) |
| R-05 | Cumplido | Sección 10: resultados, limitaciones (incl. subgrupo de fraude «suave»), riesgos, mejoras |
| R-06 | Cumplido | `scripts/validate_submission.py`: 1 columna (`score`), 113 923 filas, sin NaN/inf, orden de producción |
| R-07 | Cumplido (estimación) | AUPR de validación ≈ 0,83 (5×2 sin fuga). **Optimista**: hiperparámetros elegidos mirando los mismos pliegues |
| R-08 | Cumplido con una desviación | `requirements.txt` fijado. Objetivo Python 3.11; **verificado en 3.12** (3.11 no instalado). No se usa nada específico de 3.12 |
| D-01 | Cumplido | Mismo hash en ≥ 4 ejecuciones (script ×2, notebook en `.venv`, notebook en entorno limpio) |
| D-02 | Cumplido | Guardrail + tests; ajuste solo con construcción; `Class` de producción no influye (test) |
| D-03 | Cumplido con matiz | 8,3 min en la ejecución habitual; **15,2 min** en la ejecución limpia, con la máquina ~2× más lenta (el ajuste de `rf` pasó de 94 s a 190 s, también fuera del notebook). Sin límite de tiempo conocido (P-07) |
| D-04 | Cumplido | Traza (paso, modelo, segundos, nota) en la sección 9 |
| D-05 | Cumplido | Test con detector opcional que falla y con uno obligatorio |
| D-06 | Cumplido | `data/` y `.env` en `.gitignore`; sin CSV de datos fuera de `data/`; sin secretos; `.env.example` sin valores sensibles |
| D-07 | Cumplido | Textos, comentarios y docs en español |

## Correcciones hechas en esta fase
Al contrastar el texto del notebook con sus propias salidas se encontró que **«diferencia de medias tipificada < 0,01» era falso** (el máximo medido en el notebook es 0,0159, en `X3`); ahora dice «< 0,02». También se corrigieron los tiempos de `rf`/`et` citados, una frase sin base sobre el GMM y una afirmación sobre el AUC-ROC que el notebook no mide, y se añadió el límite principal del modelo (subgrupo de fraude «suave») a la reflexión.

## Riesgos abiertos (para conocimiento del grupo)
1. **Cabecera del CSV (P-04):** el CSV lleva la cabecera `score`. Si el evaluador lo leyese sin cabecera, la primera fila sería un texto. El enunciado no lo especifica; la decisión se confirmó con el usuario.
2. **Estimación optimista** del AUPR (ver R-07) y **sin conjunto de test independiente**.
3. **Ruido de la métrica ≈ 0,05–0,06 entre pliegues:** `hgb`, `rf`, `et` y su ensemble no son distinguibles; el ensemble se eligió por robustez.
4. **Tiempo de ejecución del notebook:** 8–15 min según la máquina.

## Cómo regenerar (si hiciese falta)
```bash
python scripts/generar_scores.py          # entrega/scores.csv (~2-4 min)
python scripts/validate_submission.py     # valida el CSV
python scripts/build_notebook.py          # lee MIEMBROS de .env; genera el notebook sin ejecutar
cd entrega && python -m nbconvert --to notebook --execute taller_anomalias.ipynb --inplace
```
