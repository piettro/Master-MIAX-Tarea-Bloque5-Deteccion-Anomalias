"""Configuración: rutas, semilla y modo demo (por variables de entorno)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def _cargar_env(ruta: Path) -> None:
    """Carga pares CLAVE=valor de un .env sin pisar el entorno existente."""
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip())


@dataclass(frozen=True)
class Config:
    """Parámetros de ejecución."""

    demo_mode: bool
    seed: int
    train_file: Path
    prod_file: Path
    target_col: str | None
    output_csv: Path
    miembros: tuple[str, ...] = ()

    @property
    def usar_demo(self) -> bool:
        """True si se pidió demo o faltan los datos reales."""
        return self.demo_mode or not (self.train_file.exists() and self.prod_file.exists())


def cargar_config() -> Config:
    """Construye la configuración desde .env / entorno."""
    _cargar_env(RAIZ / ".env")
    env = os.environ.get
    return Config(
        demo_mode=env("DEMO_MODE", "false").lower() == "true",
        seed=int(env("SEED", "42")),
        train_file=RAIZ / env("TRAIN_FILE", "data/credit_card_construction.csv"),
        prod_file=RAIZ / env("PROD_FILE", "data/credit_card_production.csv"),
        target_col=env("TARGET_COL") or None,
        output_csv=RAIZ / env("OUTPUT_CSV", "entrega/scores.csv"),
        miembros=tuple(m.strip() for m in env("MIEMBROS", "").split(";") if m.strip()),
    )
