# Detección de anomalías en transacciones bancarias

Práctica de máster (grupos). Entrega: **jueves 29/10/2026, 18:30** (aula virtual).
Entregables: `entrega/taller_anomalias.ipynb` (autocontenido) y `entrega/scores.csv` (una columna).

Documentos de referencia: [`docs/01_MATRIZ_REQUISITOS.md`](docs/01_MATRIZ_REQUISITOS.md) y [`docs/02_SPEC_TECNICA.md`](docs/02_SPEC_TECNICA.md).

## Puesta en marcha
```bash
py -3.12 -m venv .venv          # objetivo: Python 3.11
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env            # ajustar rutas/semilla; sin secretos
.venv/Scripts/python -m pytest
```

## Datos
Descargar de la carpeta de Drive del enunciado a `data/` (ignorada por git) y el notebook de partida a `notebooks/` (intacto).
Si faltan los datos o `DEMO_MODE=true`, el pipeline corre con datos sintéticos (`samples/generar_sintetico.py`).

## Estructura
`src/anomalias/{config, detectors, domain, orchestration}` · `scripts/` · `tests/` · `samples/` · `docs/` · `entrega/`.
Regla: las librerías de modelado solo se importan en `detectors/` (lo comprueba un test).

## Estado
Fases 0-6 completadas. Cierre, checksums y checklist de requisitos: `docs/05_CIERRE.md`.
- Entregables: `entrega/scores.csv` y `entrega/taller_anomalias.ipynb`.
- Decisiones y evidencias: `docs/03_DECISIONES.md`. EDA de referencia: `docs/EDA_datos.ipynb` (`pip install -r requirements-eda.txt`; regenerar con `python scripts/build_eda.py`). Validación de la Fase 6: `docs/04_VALIDACION_FASE6.md`.
- Generar scores: `python scripts/generar_scores.py` · validar: `python scripts/validate_submission.py` · notebook: `python scripts/build_notebook.py` (integrantes en `MIEMBROS` de `.env`).
