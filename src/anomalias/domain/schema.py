"""Contratos de datos: configuración de experimento y salida de detectores."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigExperimento(BaseModel):
    """Parámetros del esquema de validación."""

    n_splits: int = Field(5, ge=2)
    n_repeats: int = Field(2, ge=1)
    seed: int = 42
    columna_objetivo: str = "Class"
    permitir_produccion: bool = False  # P-05: ajustar sobre producción solo por decisión explícita


class ResultadoDetector(BaseModel):
    """Score de un detector, validado: longitud correcta, finito y sin NaN."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nombre: str
    scores: np.ndarray
    n_esperado: int

    @model_validator(mode="after")
    def _validar(self) -> "ResultadoDetector":
        s = np.asarray(self.scores, dtype=float)
        if s.ndim != 1 or len(s) != self.n_esperado:
            raise ValueError(f"{self.nombre}: longitud {s.shape} != {self.n_esperado}")
        if not np.isfinite(s).all():
            raise ValueError(f"{self.nombre}: scores con NaN o infinitos")
        self.scores = s
        return self
