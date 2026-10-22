"""Evalúa detectores y ensembles con validación sin fuga (5 folds x 2 repeticiones)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from anomalias.config import cargar_config  # noqa: E402
from anomalias.detectors.registro import ENSEMBLE_FINAL, catalogo  # noqa: E402
from anomalias.domain.features import construir_features  # noqa: E402
from anomalias.domain.schema import ConfigExperimento  # noqa: E402
from anomalias.domain.validation import deduplicar  # noqa: E402
from anomalias.orchestration.pipeline import evaluar_cv  # noqa: E402


def main() -> None:
    cfg = cargar_config()
    objetivo = cfg.target_col or "Class"
    df = deduplicar(pd.read_csv(cfg.train_file))
    y = df[objetivo].to_numpy(int)
    X, cols = construir_features(df, objetivo)
    exp = ConfigExperimento(seed=cfg.seed, columna_objetivo=objetivo)
    combos = {
        "ens_final(hgb+rf+et)": ENSEMBLE_FINAL,
        "ens_con_gmm": (*ENSEMBLE_FINAL, "gmm3"),
        "hgb+rf": ("hgb", "rf"),
    }
    res, traza = evaluar_cv(list(catalogo().values()), combos, X, y, exp, cols)
    tabla = res.groupby("modelo").agg(
        aupr=("aupr", "mean"), desv=("aupr", "std"), p_at_k=("p_at_k", "mean")
    )
    t = traza.tabla().groupby("modelo").segundos.sum().rename("seg_total")
    print(tabla.join(t).sort_values("aupr", ascending=False).round(4).to_string())
    print("\nAvisos/reintentos en traza:")
    print(traza.tabla().query("nota != 'ok'").to_string(index=False))


if __name__ == "__main__":
    main()
