# 01 · Matriz de requisitos — Práctica Detección de Anomalías

Fuente: enunciado del taller (`taller_anomalías.pdf`, páginas 1). Es la fuente de verdad junto con `02_SPEC_TECNICA.md`.

## Datos básicos
- Práctica **en grupos**.
- Entrega: **jueves 29 de octubre de 2026, 18:30**, vía aula virtual.
- Objetivo: construir un sistema para **detectar transacciones bancarias fraudulentas**.
- Datos y notebook de partida: carpeta de Google Drive indicada en el enunciado (descargar a `data/`, nunca al repo).
- Entorno: **Python**.

## Entregables y evaluación

Peso: **Resultados 75 %** · **Notebook 25 %**.

| ID | Requisito (del enunciado) | Peso | Cómo se verifica | Estado |
|---|---|---|---|---|
| R-01 | Entregar un único `.ipynb` | Notebook | Existe `entrega/taller_anomalias.ipynb` | Cumplido |
| R-02 | El notebook incluye **nombre de los componentes del grupo** | Notebook | Celda inicial con `MIEMBROS`; test que la comprueba no vacía ni placeholder | Cumplido |
| R-03 | El notebook incluye el **código desarrollado** | Notebook | Autocontenido: ejecuta de arriba abajo en entorno limpio sin importar `src/` | Cumplido |
| R-04 | El notebook incluye la **justificación de los desarrollos** | Notebook | Cada decisión técnica (features, modelo, umbral, validación) con celda markdown que la justifica y evidencia numérica | Cumplido |
| R-05 | El notebook incluye una **reflexión final** sobre los resultados | Notebook | Sección final: resultados, limitaciones, riesgos de generalización, qué haríamos con más tiempo | Cumplido |
| R-06 | Entregar un `.csv` con **una sola columna**: el score de cada fila del fichero de producción | Resultados | `scripts/validate_submission.py`: 1 columna, nº filas = fichero de producción, sin NaN/inf, orden original | Cumplido |
| R-07 | Calidad de los scores (la métrica la evalúa el profesor sobre producción) | Resultados (75 %) | Métrica interna en validación **sin fuga**; ver Preguntas abiertas P-01 | Cumplido |
| R-08 | Trabajo en Python | Ambos | `requirements.txt` fijado, Python 3.11 | Cumplido |

## Requisitos derivados (nuestros, para proteger R-06 y R-07)

| ID | Requisito | Verificación |
|---|---|---|
| D-01 | Reproducibilidad: semillas fijas, mismo CSV en dos ejecuciones | Test de determinismo con datos sintéticos |
| D-02 | Sin fuga de información: nada se ajusta con etiquetas ni datos de producción salvo que se decida explícitamente | Guardrail `leakage_guard` + tests |
| D-03 | Notebook ejecutable de extremo a extremo en tiempo razonable | `nbconvert --execute` en el cierre; tiempo registrado (objetivo orientativo ≤ 15 min, confirmar) |
| D-04 | Trazabilidad: tabla por paso (paso, modelo, segundos, nota) | Se imprime en el notebook |
| D-05 | Degradación: si falla un detector opcional, el pipeline entrega score con aviso | Test con detector que lanza excepción |
| D-06 | Sin datos ni secretos en el repo | `.gitignore` con `data/`, `.env`; revisión al cierre |
| D-07 | Idioma: README, docs, comentarios, textos del notebook y prompts en **español** | Revisión manual al cierre |

## Preguntas abiertas (resolver en la Fase 0, antes de asumir)

| ID | Pregunta | Por qué importa |
|---|---|---|
| P-01 | ¿Qué métrica puntúa el profesor (AUC-ROC, AUC-PR, precision@k, recall@k…)? ¿Está en el notebook de partida? | Define el criterio de selección de modelos |
| P-02 | ¿El fichero de entrenamiento trae etiquetas de fraude (total o parcial) o es no supervisado? | Decide supervisado vs. no supervisado vs. mixto |
| P-03 | ¿Convención del score: mayor = más anómalo? | Un signo invertido puede hundir el 75 % |
| P-04 | ¿El CSV lleva cabecera? ¿Índice? | El enunciado dice "una sola columna" |
| P-05 | ¿Está permitido ajustar (sin etiquetas) sobre el fichero de producción (transductivo)? | Afecta a la validez del enfoque |
| P-06 | Nombres de los integrantes del grupo | Requisito R-02 |
| P-07 | ¿Hay restricciones de librerías o límite de tiempo de ejecución? | Elección de modelos |
| P-08 | El notebook de partida: ¿qué debe respetarse de su estructura? | Evitar penalización de formato |

## Fuera de alcance
Despliegue/API, interfaz web, LLMs externos, bases de datos, redes profundas salvo que la validación demuestre que superan a la alternativa clásica y quepan en tiempo.

## Requisitos de la ampliación (EDA)

| ID | Requisito | Verificación | Estado |
|---|---|---|---|
| E-01 | Notebook de EDA detallado que sirva de referencia para las decisiones | `docs/EDA_datos.ipynb` ejecuta sin errores; mapa «hallazgo → decisión» calculado; 22 afirmaciones del texto verificadas automáticamente (sec. 12b) | Cumplido (v2) |
| E-02 | Fase 5: mejorar el EDA | Revisión afirmación por afirmación; ejecución determinista (textos y figuras idénticos en 2 ejecuciones) | Cumplido |
| E-03 | Fase 6: validar que el proyecto es coherente con las conclusiones del EDA | Contraste de H1…H10 con el pipeline real (`docs/04_VALIDACION_FASE6.md`) | Hecha, pendiente de confirmación |
| E-04 | El EDA no incrusta datos ni se entrega como parte del CSV/notebook | Test `test_eda_no_incrusta_datos_en_las_celdas` | Cumplido |
