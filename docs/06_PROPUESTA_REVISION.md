# 06 · Propuesta de revisión del notebook

Rama `propuesta/revision-notebook`. El notebook revisado está en `propuesta/taller_anomalias.ipynb` (con su `propuesta/scores.csv`). `entrega/` y los checksums de `docs/05_CIERRE.md` no se han tocado.

## Qué cambia y por qué
1. **Datos**: si faltan los CSV, el notebook se detiene con un error claro (fichero, carpetas buscadas, opciones). Antes pasaba a datos sintéticos de forma silenciosa. Se admite `.zip` y la variable `DESCARGAR_DATOS=true` (descarga de Drive, pensada para Colab). El modo demo solo se activa con `DEMO_MODE=true`.
2. **Cifras que cuadran**: se eliminan las cifras de la referencia 5x2 que el notebook no calcula. El texto cita la ejecución 5x1 (determinista con `SEED=42`). Una celda de comprobación contrasta cada cifra citada con el valor calculado y **avisa y continúa** si alguna no cuadra (con `COMPROBACION_ESTRICTA=true` detiene el notebook; en demo se omite). Las cifras salen de un único diccionario (`CIFRAS` en `scripts/build_notebook.py`).
3. **Baseline recalculado**: el GMM(5) sobre variables crudas del notebook de partida se evalúa dentro del notebook con las mismas particiones (AUPR medio por fold 0,375).
4. **Gráficos** (matplotlib): curva precisión-recall fuera de pliegue, distribución del score de producción e importancia por permutación de `hgb`. `evaluar_cv` admite un parámetro opcional `oof` para recoger los scores fuera de pliegue; no cambia su retorno.
5. **Subgrupos de fraude** calculados dentro del notebook (KMeans descriptivo, recall@k por grupo, parte del fraude no detectado que es "suave"), en lugar de citar un análisis externo.
6. **Redacción**: sin primera persona ni guiones largos.
7. **Legibilidad**: tabla "Mapa del notebook" y celdas de código de apoyo plegadas (metadato `source_hidden`).
8. **Dependencia nueva**: `matplotlib==3.11.2` en `requirements.txt`.

## Cómo regenerar
```
python scripts/build_notebook.py --miembros "Piettro Rodrigues;Alonso Díaz;Raúl Rodríguez" --salida propuesta/taller_anomalias.ipynb
cd propuesta
python -m nbconvert --to notebook --execute taller_anomalias.ipynb --inplace --ExecutePreprocessor.timeout=3600
```
Con los CSV en `data/` y el entorno de `requirements.txt`. La ejecución genera `propuesta/scores.csv`.

## Resultado de validación
- SHA-256 de `propuesta/scores.csv`: `8b5371a1a93ac4bd9c87c7c46b00b458c9800786d37ea43353bf1229d1cdd257`, **idéntico** al de `entrega/scores.csv`. Modelo y scoring no cambian.
- La celda de comprobación de cifras pasa con datos reales.
- `pytest -q` en verde (el test del notebook usa el modo demo).

## Pendiente
Decisión de Piettro: pasar el notebook a `entrega/` (sustituyendo `entrega/taller_anomalias.ipynb`), volver a calcular su SHA-256 y actualizar la tabla de `docs/05_CIERRE.md`. Otros textos que mencionan el comportamiento anterior (por ejemplo, el README sobre la caída a demo) no se han revisado.

## Ajustes tras la revisión (rama `revision/ajustes-pr1`)
Revisión técnica de la PR 1 y ajustes aplicados sobre ella. `entrega/` y los checksums de `docs/05_CIERRE.md` siguen sin tocarse.
1. **Conflicto de `matplotlib`**: `requirements.txt` fija 3.11.2 y `requirements-eda.txt` fijaba 3.10.8 con `-r requirements.txt`; `pip` rechazaba instalarlo. Se elimina la línea duplicada. El EDA se ejecuta sin errores con 3.11.2 y sus 22 afirmaciones se cumplen.
2. **Comprobación de cifras no fatal por defecto**: la tolerancia de ±0,0005 y el `assert` hacían frágil el notebook entregable ante otras versiones de librerías o sistemas (varias cifras estaban cerca del límite, p. ej. `recall_fuerte` 0,99565 frente a 0,996). Ahora imprime un `AVISO` con las cifras que difieren y continúa; `COMPROBACION_ESTRICTA=true` restaura la parada. La comparación es `comparar_cifras` (en `domain/diagnostico.py`) y la celda tiene tests para los modos aviso, estricto y demo.
3. **`propuesta/scores.csv` deja de versionarse** (2,3 MB duplicado). Su SHA-256 queda documentado abajo y en `.gitignore`.
4. **Documentación**: excepción a la regla 3 de la spec registrada (`docs/02`, `docs/03`) y README actualizado.

## Validación de los ajustes
- `pytest`: **54 passed** (49 de la PR + 5 nuevos de la comprobación de cifras).
- **Entorno limpio** (venv nuevo con solo `requirements.txt`; numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, Python 3.12): el notebook regenerado ejecuta **sin errores en 575 s**, con 3 figuras y «Todas las cifras citadas coinciden con lo calculado».
- `scores.csv` generado: SHA-256 `8b5371a1a93ac4bd9c87c7c46b00b458c9800786d37ea43353bf1229d1cdd257`, **idéntico** a `entrega/scores.csv`.
- `propuesta/taller_anomalias.ipynb` es la versión ejecutada en ese entorno limpio.
- No probado: la descarga real desde Drive y Python 3.11.

## Estado: adoptado (27/10/2026)
Piettro aceptó la propuesta. `propuesta/taller_anomalias.ipynb` se copió a `entrega/taller_anomalias.ipynb` (SHA-256 `248b5ffeaacbb7739e8ea227aeb0405d3e8240c92984f9f260c6c59907d7f051`), el hash se recalculó en `docs/05_CIERRE.md` (versión 2) y los textos externos que citaban el baseline 0,44 pasaron a 0,375. `entrega/scores.csv` no cambia. `propuesta/` queda como copia idéntica del notebook; si se retoca el notebook, regenerar `entrega/` y actualizar el cierre.
