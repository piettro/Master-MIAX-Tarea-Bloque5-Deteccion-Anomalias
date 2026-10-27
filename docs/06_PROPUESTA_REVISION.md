# 06 · Propuesta de revisión del notebook

Rama `propuesta/revision-notebook`. El notebook revisado está en `propuesta/taller_anomalias.ipynb` (con su `propuesta/scores.csv`). `entrega/` y los checksums de `docs/05_CIERRE.md` no se han tocado.

## Qué cambia y por qué
1. **Datos**: si faltan los CSV, el notebook se detiene con un error claro (fichero, carpetas buscadas, opciones). Antes pasaba a datos sintéticos de forma silenciosa. Se admite `.zip` y la variable `DESCARGAR_DATOS=true` (descarga de Drive, pensada para Colab). El modo demo solo se activa con `DEMO_MODE=true`.
2. **Cifras que cuadran**: se eliminan las cifras de la referencia 5x2 que el notebook no calcula. El texto cita la ejecución 5x1 (determinista con `SEED=42`). Una celda de comprobación contrasta cada cifra citada con el valor calculado y detiene el notebook si no cuadra (en demo se omite). Las cifras salen de un único diccionario (`CIFRAS` en `scripts/build_notebook.py`).
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
