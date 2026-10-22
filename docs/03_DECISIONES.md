# 03 · Bitácora de decisiones

## Fase 0 — Preguntas abiertas (confirmadas por el usuario)
| ID | Decisión | Evidencia |
|---|---|---|
| P-01 | Métrica: AUPR (`average_precision_score`) | Notebook de partida, celda 1 |
| P-02 | Hay etiquetas → enfoque supervisado + no supervisado | 295 positivos de 170 884 (0,173 %) |
| P-03 | Mayor score = más fraude | Notebook de partida |
| P-04 | CSV con cabecera `score`, sin índice | Propuesta (enunciado no lo aclara) |
| P-05 | No ajustar sobre producción | Deriva KS ≤ 0,005; hay etiquetas |
| P-06 | Integrantes | **Pendiente** (llegó `<nombres>` sin rellenar) |
| P-07 | Solo sklearn | Sin restricciones en el notebook |
| P-08 | Respetar AUPR y convención de score; rutas locales en vez de Colab | Notebook de partida |

## Fase 1 — Validación
- Deduplicar construcción antes de partir: 170 884 → 166 717 filas, 295 → 285 positivos (evita copias entre ajuste y validación).
- K-fold estratificado 5×2 con semillas fijas (~57 positivos por fold de validación).
- AUPR implementada en numpy (regla 2: sklearn solo en `detectors/`); test de igualdad con sklearn incluyendo empates.
- Todo ajuste pasa por `ajustar_seguro` (guardrail).

## Baselines (5 folds × 2 repeticiones, sin fuga)
| Modelo | AUPR media | std | P@k | s (total) |
|---|---|---|---|---|
| GMM(5) solo normales (notebook) | 0,436 | 0,080 | 0,447 | 68 |
| Isolation Forest | 0,215 | 0,048 | 0,312 | 9 |
| HistGradientBoosting (crudo) | 0,076 | 0,108 | 0,153 | 15 |

Lectura: la densidad sobre filas normales gana; el GBM crudo rinde mal y es inestable (pocos positivos, colas pesadas): se investiga en la Fase 2.

## Fase 2 — Variables, detectores y ensemble
Evidencia (AUPR media, 5 folds × 2 repeticiones sin fuga, 166 717 filas, 285 positivos):

| Modelo | AUPR | desv. entre folds | P@k |
|---|---|---|---|
| **ens_final = rango medio(hgb, rf, et)** | **0,834** | 0,050 | 0,830 |
| hgb + rf | 0,832 | 0,051 | 0,828 |
| hgb (regularizado) | 0,831 | 0,046 | 0,828 |
| rf (balanceado) | 0,822 | 0,054 | 0,823 |
| et (balanceado) | 0,819 | 0,049 | 0,812 |
| ensemble con GMM | 0,805 | 0,051 | 0,802 |
| GMM(3) sobre variables transformadas | 0,707 | 0,060 | 0,784 |
| Isolation Forest (solo normales) | 0,207 | 0,056 | 0,323 |
| *Baseline Fase 1: GMM(5) crudo / HGB por defecto* | *0,436 / 0,076* | | |

Decisiones y por qué:
1. **Transformación asinh(x/100) y log1p(importe) — CORRECCIÓN (Fase 3)**: la nota original decía que subía el GMM de 0,373 a 0,743, pero esa comparación mezclaba el cambio de 5 a 3 componentes. Medido a igualdad de modelo (GMM(3), misma partición, ejecución real del notebook): crudo 0,750 frente a transformado 0,732; es decir, **no mejora**. Los árboles son invariantes (HGB crudo = transformado = 0,836). Se mantiene como higiene (inocua para el ensemble final, que son árboles), no como mejora demostrada. La mejora real del baseline GMM(5) → GMM(3) viene del número de componentes (GMM(5) crudo es inestable).
2. **Regularizar el boosting es imprescindible**: con parámetros por defecto (lr 0,1, l2 = 0, hojas 20) el AUPR es 0,076–0,108 y muy inestable; con l2 = 10, min_samples_leaf = 50, 8 hojas y lr 0,05 sube a ≈ 0,83–0,85.
3. **El GMM no aporta al ensemble** (0,805 con él frente a 0,834 sin él): se mantiene como detector candidato opcional, fuera del ensemble final. Las pruebas con peso 0,1–0,5 también empeoraron (3 folds).
4. **Ensemble hgb+rf+et por rangos**: mejor media, pero la ganancia sobre hgb solo (+0,003) está muy dentro del ruido (desv. entre folds ≈ 0,05). Se elige por reducir varianza, no por una mejora demostrada.
5. **Coste**: hgb ≈ 1 s por ajuste; rf ≈ 70 s y et ≈ 34 s (con 2 hilos). La evaluación completa tarda ≈ 10 min.
6. Rangos calculados sobre el propio vector de scores a puntuar: no ajustan parámetros con producción, solo ordenan.

## Fase 3 — Scoring, CSV y notebook
- **Ajuste**: construcción deduplicada completa (166 717 filas); producción solo se puntúa (P-05). `verificar_ajuste_no_produccion` bloquea que los datos de ajuste sean los de producción.
- **CSV**: `entrega/scores.csv`, cabecera `score`, sin índice, 113 923 filas, en el orden de producción (P-04 pendiente de confirmar por el profesor/enunciado).
- **Determinismo (D-01)**: dos ejecuciones independientes de `scripts/generar_scores.py` y una ejecución completa del notebook sobre datos reales dan el mismo SHA-256.
- **Notebook**: se genera con `scripts/build_notebook.py` (inlinea `src/`, sin `import anomalias`). Se rechaza generar la versión final si faltan `MIEMBROS` (R-02). Ejecución real de arriba abajo ≈ 8 min (CV 5×1: 384 s).
- **Evidencia con la ejecución real (partición 0)**: HGB por defecto 0,002 frente a regularizado 0,836; CV 5×1: ensemble 0,834 · hgb 0,834 · rf 0,823 · et 0,820 · con GMM 0,806 · gmm3 0,706 · iforest 0,194.

## EDA (v1) — hallazgos que respaldan o matizan decisiones
Detalle y cifras en `docs/EDA_datos.ipynb`. Resumen:
- **Respaldan**: deduplicar (4 167 copias; 16 filas de fraude en grupos repetidos); AUPR y estratificación (prevalencia 0,171 %); enfoque supervisado (12 de 21 variables con AUC ≥ 0,8, hasta 0,95; diferencias de hasta 8σ); sin deriva producción–construcción (AUC adversaria 0,500) → no ajustar con producción.
- **Matizan / corrigen**: (1) las X son **linealmente incorrelacionadas pero no independientes** (Spearman hasta 0,44; magnitudes hasta 0,73); (2) el fraude es mucho más atípico que lo normal, y un Mahalanobis sobre variables transformadas llega a AUPR ≈ 0,69 sin etiquetas, pero con X en bruto solo 0,26 → la transformación es crítica para modelos de una gaussiana, irrelevante para árboles; (3) en la PCA el fraude **no** está en la periferia sino en la zona central-derecha; (4) dos subpoblaciones de fraude plausibles (silueta k=2 ≈ 0,40; no concluyente con 285 positivos).
- **Hallazgos nuevos**: `Amount = 0` es 6,5× más frecuente en fraude; el 34 % de los fraudes comparte solo 3 importes exactos (1,50 · 149,98 · 1,14) y 149,98 tiene una tasa de fraude 56× la global; la mediana de variables extremas (|z|>3) por fila es 7 en fraude y 0 en normales.
- **Errores propios corregidos durante el EDA**: se afirmó que un score de rareza "separa bien en AUC pero mal en AUPR" (falso para Mahalanobis transformado: 0,69) y que el fraude se desplaza a la periferia de la nube PCA (falso). Ambas afirmaciones se verificaron contra las cifras y se reescribieron.
- **Hipótesis abiertas para la Fase 6**: H1…H8 (sección 13 del EDA).

## Fase 5 — EDA v2 (mejora del notebook de EDA)
**Errores de la v1 detectados y corregidos** (revisión de figuras y de cada afirmación contra las cifras):
- Figura de clases: la barra "aleatorio" (AUPR 0,0017) era invisible → escala log con etiquetas.
- Histograma de `Amount` "recortado en p99" creaba un pico artificial en 1 500 → se muestran solo valores ≤ p99.
- Histogramas de variables discriminantes recortados en los percentiles 0,2/99,8 daban la falsa impresión de "amontonamiento en el borde": en realidad el fraude forma un **modo propio más allá** de ese recorte. Ahora se muestra el rango completo y se cuantifica con una tabla de cola.
- Texto de la sección 8/9/10 ya corregido en la v1 (PCA, silueta, deriva); se añadió una verificación automática para que no pueda reaparecer una discrepancia texto/datos.

**Análisis nuevos**: firma de cola por variable (hasta 74 % de los fraudes por debajo del p0,2 en `X17` = 368× lo esperado; 10 variables con ≥ 30 %); orden de filas no informativo (KS p = 0,84); subgrupos de fraude estables (ARI ≈ 1,0 entre semillas, 0,95 con bootstrap): 227 "fuertes" (desplazamiento medio 4,1σ) y 68 "suaves" (0,8σ); `149,98` solo en el grupo fuerte; frecuencia del importe = señal modesta (0,6×–2,0×); IC95 % del AUC (Hanley-McNeil; 10 variables con límite inferior ≥ 0,80); dependencia no lineal ilustrada (hexbin); meseta superior de `X1` (11 % de filas, ningún fraude).

**Rigor**: 22 afirmaciones numéricas del texto se contrastan en la celda 12b y el notebook se detiene si alguna falla. La ejecución es determinista (texto y figuras idénticos en dos ejecuciones). Las 9 hipótesis (H1…H9) indican ahora qué resultado las refutaría.

**Cautela declarada**: el recuento de colas (AUPR ≈ 0,56) elige variables con las etiquetas en la misma muestra; es una cota optimista, no validación.

## Fase 6 — Validación del proyecto contra el EDA
Informe completo y reproducible: `docs/04_VALIDACION_FASE6.md` (`scripts/validar_fase6.py`). Esquema 5×2 sin fuga, comparaciones pareadas, criterio conservador de mejora real (Δ > 0, z ≥ 2, gana en ≥ 80 % de particiones). **El pipeline y los entregables no se han modificado** (`scores.csv` conserva su SHA-256).
- **Conclusión**: el proyecto es coherente con el EDA. Confirmadas H3 (transformación neutra para árboles), H6 (el modelo detecta 99 % del fraude "fuerte" y solo 29 % del "suave"; el 97 % de lo no detectado es suave), H7 (ensemble ≈ hgb), H8 (producción ≈ validación: KS ≤ 0,06; ~187 filas sobre el umbral frente a ~195 esperadas) y H2 (poco material: +0,008 ≪ ruido). Parcial H1. Refutada H5.
- **H4 (frecuencia del importe)**: exploración HGB +0,0065 (z = 3,0); confirmación en particiones nuevas sobre el ensemble real +0,0042 (z = 1,6; gana en 4/5; mejora los tres modelos). Indicio consistente pero pequeño (~10× menor que el ruido): **no se aplica sin decisión del usuario**.
- **H10 (binning de HGB en la cola)**: variables "distancia a la cola" +0,004 en exploración y +0,004 en confirmación, sin alcanzar el criterio. No se aplica.
- **Matiz al EDA**: la importancia por permutación concentra el modelo en `X14` (0,62), pero quitarla solo cuesta 0,011 de AUPR: la señal es redundante entre variables.
- **Código añadido** (no usado por el pipeline entregado): `domain/diagnostico.py`, `detectors/aumentado.py` (variables derivadas calculadas solo con filas de ajuste) y `tests/test_fase6.py`.

**Decisión del usuario (tras la Fase 6):** el proyecto se queda **como está**, sin la frecuencia del importe (H4). `entrega/scores.csv` y `entrega/taller_anomalias.ipynb` no cambian. Motivo: la ganancia (+0,004 en el ensemble) es ≈ 10 veces menor que el ruido entre pliegues y no alcanzó el criterio de confirmación.
