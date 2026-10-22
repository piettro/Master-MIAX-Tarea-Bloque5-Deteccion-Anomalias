"""Valida el CSV de entrega contra el fichero de producción (R-06)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from anomalias.config import cargar_config  # noqa: E402
from anomalias.orchestration.submission import validar_csv  # noqa: E402


def main() -> None:
    cfg = cargar_config()
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.output_csv
    n = len(pd.read_csv(cfg.prod_file, usecols=[0]))
    problemas = validar_csv(ruta, n)
    print(f"{ruta} ({n} filas esperadas):", "VÁLIDO" if not problemas else problemas)
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    main()
