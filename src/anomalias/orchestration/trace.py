"""Traza de ejecución: (paso, modelo, segundos, nota)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd


@dataclass
class Traza:
    """Acumula pasos; se muestra como tabla en el notebook."""

    filas: list[dict] = field(default_factory=list)

    def registrar(self, paso: str, modelo: str, segundos: float, nota: str = "") -> None:
        self.filas.append(
            {"paso": paso, "modelo": modelo, "segundos": round(segundos, 3), "nota": nota}
        )

    @contextmanager
    def medir(self, paso: str, modelo: str = "-", nota: str = "") -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.registrar(paso, modelo, time.perf_counter() - t0, nota)

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas, columns=["paso", "modelo", "segundos", "nota"])
