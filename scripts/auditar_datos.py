"""Auditoría descriptiva de data/ (sin ajustar ningún modelo)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from anomalias.config import cargar_config  # noqa: E402


def resumen(nombre: str, df: pd.DataFrame) -> None:
    """Imprime forma, tipos, nulos, duplicados y cardinalidades."""
    print(f"\n=== {nombre}: {df.shape[0]} filas x {df.shape[1]} columnas ===")
    tabla = pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "nulos": df.isna().sum(), "cardinalidad": df.nunique()}
    )
    print(tabla.to_string())
    print(f"Duplicados: {df.duplicated().sum()}")
    for col in df.columns:
        if "date" in col.lower() or "fecha" in col.lower() or "time" in col.lower():
            f = pd.to_datetime(df[col], errors="coerce")
            print(f"Rango temporal {col}: {f.min()} -> {f.max()}")


def main() -> None:
    cfg = cargar_config()
    ficheros = {"entrenamiento": cfg.train_file, "producción": cfg.prod_file}
    faltan = [str(p) for p in ficheros.values() if not p.exists()]
    if faltan:
        print("No se encuentran los datos reales:", *faltan, sep="\n  - ")
        sys.exit(1)
    dfs = {k: pd.read_csv(v) for k, v in ficheros.items()}
    for k, df in dfs.items():
        resumen(k, df)
    tr, pr = (set(d.columns) for d in dfs.values())
    print("\nSolo en entrenamiento:", sorted(tr - pr))
    print("Solo en producción:", sorted(pr - tr))


if __name__ == "__main__":
    main()
