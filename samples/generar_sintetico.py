"""Transacciones sintéticas con fraude inyectado (modo demo y tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generar_sintetico(n: int = 2000, tasa_fraude: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Genera n transacciones; `es_fraude` marca las inyectadas (importes y horas atípicos)."""
    rng = np.random.default_rng(seed)
    n_fraude = max(1, int(n * tasa_fraude))
    es_fraude = np.zeros(n, dtype=int)
    es_fraude[rng.choice(n, n_fraude, replace=False)] = 1
    f = es_fraude == 1

    importe = rng.lognormal(3.5, 0.8, n)
    importe[f] = rng.lognormal(6.0, 0.7, n_fraude)
    hora = np.clip(rng.normal(14, 4, n), 0, 23).round()
    hora[f] = rng.choice([1, 2, 3, 4], n_fraude)
    distancia_km = rng.exponential(10, n)
    distancia_km[f] = rng.exponential(200, n_fraude)
    n_ult_24h = rng.poisson(1.5, n)
    n_ult_24h[f] = rng.poisson(6, n_fraude)
    return pd.DataFrame(
        {
            "importe": importe,
            "hora": hora,
            "distancia_km": distancia_km,
            "n_ult_24h": n_ult_24h,
            "es_fraude": es_fraude,
        }
    )
