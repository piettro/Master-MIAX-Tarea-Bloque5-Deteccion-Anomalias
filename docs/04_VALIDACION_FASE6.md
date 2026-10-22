# 04 · Validación de la Fase 6: ¿es coherente el proyecto con las conclusiones del EDA?

Generado por `scripts/validar_fase6.py` (etapa `informe`). Cifras calculadas con el pipeline real, esquema **5 folds × 2 repeticiones sin fuga**, datos deduplicados (166.717 filas, 285 fraudes), AUPR como métrica.

## Metodología y cautelas
- **Ruido de referencia**: desviación entre particiones del AUPR de HGB = 0,0459. Las comparaciones son **pareadas** por partición.
- **Criterio de mejora real** (conservador): diferencia media > 0, z ≥ 2 y ganar en ≥ 80 % de las particiones. Las 10 particiones comparten datos (2 repeticiones), por lo que el z está algo inflado.
- **Sin fuga**: todo ajuste pasa por el guardrail; las variables derivadas (colas, importes clave, frecuencia) se calculan **solo con las filas de ajuste** (`AumentoDerivadas`).
- **Comparaciones múltiples**: en la exploración se probaron 5 variantes sobre las mismas particiones; por eso la única candidata se **re-evalúa en particiones nuevas** sobre el ensemble real.
- El pipeline entregado **no se ha modificado** en esta fase.

## Resumen

| Hipótesis | Veredicto | ¿Cambia el proyecto? |
|---|---|---|
| H1 | Parcial | No |
| H2 | Confirmada, pero poco material | No (deduplicar se mantiene por corrección) |
| H3 | Confirmada para HGB; también para Extra Trees | No |
| H4 | Indicio consistente de mejora pequeña; no alcanza el criterio | No (decisión del usuario: se deja el proyecto como está) |
| H5 | Refutada | No |
| H6 | Confirmada | No (es el límite principal del modelo; con estas variables no hay evidencia de cómo superarlo) |
| H7 | Confirmada (diferencia no significativa) | No |
| H8 | Confirmada | No |
| H10 | No confirmada | No |
| H9 | Resuelta en el EDA | No |

## Detalle

### H1 — Parcial
`X13` y `X15` no aportan (importancia -0,0012 y 0,0006; quitarlas: Δ = 0,0016 (e.t. 0,0021; z = 0,8; gana en 7/10)). La importancia concuerda solo moderadamente con el AUC univariante (Spearman 0,58) y está **muy concentrada en `X14`** (0,623 de caída de AUPR al permutarla). Pese a ello, **quitar `X14` solo cuesta 0,0110 de AUPR** (0,8200 frente a 0,8310; Δ = -0,0110 (e.t. 0,0041; z = -2,7; gana en 2/10)): la señal está repartida y es redundante en otras variables, como indicaba el EDA; la importancia por permutación exagera la dependencia. (Un HGB con `X14` sola rinde 0,2890.)
**¿Cambia el proyecto?** No

### H2 — Confirmada, pero poco material
Sin deduplicar el AUPR sube de 0,8310 a 0,8394 (+0,0084), mucho menos que el ruido entre pliegues (0,0459). Aun así, el 3,4 % de las filas de validación tiene una copia exacta en ajuste (0 % tras deduplicar).
**¿Cambia el proyecto?** No (deduplicar se mantiene por corrección)

### H3 — Confirmada para HGB; también para Extra Trees
HGB: resultado idéntico con X en bruto y transformadas (idéntico en las 10 particiones). Extra Trees (umbrales aleatorios): crudo 0,8188 frente a transformado 0,8198 (Δ = -0,0010 (e.t. 0,0075; z = -0,1; gana en 2/5)).
**¿Cambia el proyecto?** No

### H4 — Indicio consistente de mejora pequeña; no alcanza el criterio
El indicador de importes clave no aporta; la **frecuencia del importe** mejoró HGB en la exploración. importes: 0,8308 (Δ = -0,0002 (e.t. 0,0003; z = -0,9; gana en 2/10)); frecuencia: 0,8375 (Δ = 0,0065 (e.t. 0,0022; z = 3,0; gana en 9/10)). **Confirmación en particiones nuevas (semilla distinta), sobre el ensemble real:** base 0,8394 → con frecuencia 0,8436 (Δ = 0,0042 (e.t. 0,0026; z = 1,6; gana en 4/5)); hgb: Δ = 0,0077 (e.t. 0,0039; z = 2,0; gana en 4/5); rf: Δ = 0,0068 (e.t. 0,0060; z = 1,1; gana en 4/5); et: Δ = 0,0032 (e.t. 0,0015; z = 2,1; gana en 4/5)
**¿Cambia el proyecto?** No por ahora (ganancia ≈ 10 veces menor que el ruido entre pliegues); **opcional si el usuario lo decide**

### H5 — Refutada
Añadir el recuento de colas tocadas: 0,8302 frente a 0,8310 (Δ = -0,0008 (e.t. 0,0015; z = -0,6; gana en 4/10)). Los árboles ya capturan esa información.
**¿Cambia el proyecto?** No

### H6 — Confirmada
Recall entre los k primeros (k = nº de fraudes) del ensemble: grupo fuerte 99,4 % (n=218) frente a grupo suave 29,4 % (n=67). Percentil medio del score: 99,93 % frente a 89,53 %. **El 97 % de los fraudes que el modelo no coloca entre los k primeros pertenece al grupo suave**: el techo de AUPR ≈ 0,83 es, en esencia, el techo de detección del fraude suave.
**¿Cambia el proyecto?** No (es el límite principal del modelo; con estas variables no hay evidencia de cómo superarlo)

### H7 — Confirmada (diferencia no significativa)
Ensemble − hgb: Δ = 0,0026 (e.t. 0,0040; z = 0,6; gana en 7/10). Ensemble − rf: Δ = 0,0120 (e.t. 0,0042; z = 2,9; gana en 9/10). Ensemble − et: Δ = 0,0142 (e.t. 0,0059; z = 2,4; gana en 8/10). hgb − rf: Δ = 0,0094 (e.t. 0,0053; z = 1,8; gana en 8/10). El ensemble supera de forma consistente a `rf` y `et` por separado, pero **no se distingue de `hgb` solo**: `hgb` bastaría; el ensemble aporta robustez, no una mejora demostrada.
**¿Cambia el proyecto?** No

### H8 — Confirmada
hgb: KS 0,013, 189 filas de producción sobre el umbral que en validación deja pasar un 0,171 % (esperadas ≈ 195); rf: KS 0,001, 185 filas de producción sobre el umbral que en validación deja pasar un 0,171 % (esperadas ≈ 195); et: KS 0,060, 187 filas de producción sobre el umbral que en validación deja pasar un 0,171 % (esperadas ≈ 195)
**¿Cambia el proyecto?** No

### H10 — No confirmada
Hipótesis surgida en esta fase: con una sola variable, HGB rinde 0,2890 mientras que el score directo `-X14` da ≈ 0,63 (EDA), lo que sugiere que el *binning* por cuantiles pierde resolución en la cola. Variables 'distancia a la cola': exploración 0,8310 → 0,8350 (Δ = 0,0040 (e.t. 0,0028; z = 1,4; gana en 7/10)); confirmación en particiones nuevas 0,8284 → 0,8322 (Δ = 0,0038 (e.t. 0,0042; z = 0,9; gana en 6/10)). La dirección es siempre positiva pero pequeña y no alcanza el criterio; la explicación del *binning* queda como indicio no probado.
**¿Cambia el proyecto?** No

### H9 — Resuelta en el EDA
KS de posición p = 0,84; sin autocorrelación.
**¿Cambia el proyecto?** No

## Conclusiones
1. **El proyecto es coherente con el EDA**: la deduplicación, la transformación (neutra para árboles), la ausencia de ajuste con producción, el uso de árboles y la métrica quedan respaldados; ninguna conclusión del EDA obliga a cambiar el pipeline.
2. **El límite del modelo ya está explicado (H6):** casi todo el fraude no detectado es del grupo "suave". Mejorar el AUPR pasaría por detectar ese grupo, algo que estas variables apenas permiten.
3. **Hipótesis del EDA refutadas o sin efecto medible** (H5, H10, y H4 en parte): los árboles ya extraen esa información. Se evitó, así, añadir complejidad sin evidencia.
4. **Única mejora candidata: la frecuencia del importe (H4)**, con ganancia consistente pero pequeña (≈ +0,004 en el ensemble, mejora los tres modelos en particiones nuevas) que no alcanza el criterio conservador. **No se aplica sin tu decisión**; si se aplicara habría que regenerar el CSV y el notebook y revalidar.
5. **Ensemble frente a `hgb` solo (H7):** equivalentes dentro del ruido; se mantiene el ensemble por robustez, como se decidió en la Fase 2.

## Reproducción
```bash
python scripts/validar_fase6.py oof        # ~15 min
python scripts/validar_fase6.py produccion
python scripts/validar_fase6.py rapidos
python scripts/validar_fase6.py x14
python scripts/validar_fase6.py et          # ~6 min
python scripts/validar_fase6.py finas
python scripts/validar_fase6.py confirmar   # ~10 min
python scripts/validar_fase6.py analisis
python scripts/validar_fase6.py informe
```
