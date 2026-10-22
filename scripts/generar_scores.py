"""Genera entrega/scores.csv: ajusta con construcción y puntúa producción."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))
from anomalias.config import cargar_config  # noqa: E402
from anomalias.detectors.registro import ENSEMBLE_FINAL, catalogo  # noqa: E402
from anomalias.domain.validation import deduplicar  # noqa: E402
from anomalias.orchestration.submission import (  # noqa: E402
    entrenar_y_puntuar, escribir_scores, validar_csv,
)


def main() -> None:
    cfg = cargar_config()
    if cfg.usar_demo:
        from samples.generar_sintetico import generar_sintetico

        objetivo, importe = "es_fraude", "importe"
        train, prod = generar_sintetico(6000, seed=cfg.seed), generar_sintetico(2000, seed=cfg.seed + 1)
        prod[objetivo] = float("nan")
        salida = cfg.output_csv.with_name(cfg.output_csv.stem + "_demo.csv")
        print("MODO DEMO: datos sintéticos")
    else:
        objetivo, importe = cfg.target_col or "Class", "Amount"
        train, prod = pd.read_csv(cfg.train_file), pd.read_csv(cfg.prod_file)
        salida = cfg.output_csv
    train = deduplicar(train)
    cat = catalogo()
    especs = [cat[n] for n in ENSEMBLE_FINAL]
    scores, res = entrenar_y_puntuar(especs, train, prod, objetivo, importe, cfg.seed)
    escribir_scores(scores, salida)
    print(res.traza.tabla().to_string(index=False))
    for aviso in res.avisos:
        print("AVISO:", aviso)
    problemas = validar_csv(salida, len(prod))
    print(f"{salida}: {'VÁLIDO' if not problemas else problemas}")
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    main()
