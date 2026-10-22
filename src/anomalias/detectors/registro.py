"""Catálogo de detectores disponibles y composición del ensemble final."""
from __future__ import annotations

from anomalias.detectors.base import EspecDetector
from anomalias.detectors.bosques import ExtraTreesDetector, RandomForestDetector
from anomalias.detectors.gbm_supervisado import GBMSupervisado
from anomalias.detectors.gmm import GMMDetector
from anomalias.detectors.isolation_forest import IsolationForestDetector

# Ensemble final (decisión con evidencia en docs/03_DECISIONES.md)
ENSEMBLE_FINAL: tuple[str, ...] = ("hgb", "rf", "et")


def catalogo() -> dict[str, EspecDetector]:
    """Todos los detectores candidatos; hgb es obligatorio, el resto opcionales."""
    especs = [
        EspecDetector("hgb", lambda s: GBMSupervisado(s), opcional=False),
        EspecDetector("rf", lambda s: RandomForestDetector(s)),
        EspecDetector("et", lambda s: ExtraTreesDetector(s)),
        EspecDetector("gmm3", lambda s: GMMDetector(3, s)),
        EspecDetector("iforest", lambda s: IsolationForestDetector(seed=s)),
    ]
    return {e.nombre: e for e in especs}
