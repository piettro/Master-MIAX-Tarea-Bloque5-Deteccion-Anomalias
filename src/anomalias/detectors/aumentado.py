"""Envoltorio experimental (hipótesis H4/H5 del EDA): añade variables derivadas calculadas SOLO con las filas de ajuste."""
from __future__ import annotations

import numpy as np

from anomalias.detectors.base import Detector


class AumentoDerivadas:
    """Añade al detector base: recuento de colas tocadas, indicadores de importes clave y frecuencia del importe.

    Todo (percentiles, variables y sentido de cada cola, importes clave, frecuencias) se aprende en `fit`
    con las filas de ajuste; `score` solo aplica lo aprendido. La columna de importe se espera como log1p(importe).
    """

    def __init__(
        self,
        base: Detector,
        col_importe: int,
        colas: bool = False,
        importes: bool = False,
        frecuencia: bool = False,
        colas_finas: bool = False,
        q: float = 0.002,
        min_frac_cola: float = 0.10,
        min_fraudes_importe: int = 3,
        min_razon_importe: float = 5.0,
    ) -> None:
        self.base = base
        self.col_importe = col_importe
        self.colas, self.importes, self.frecuencia, self.colas_finas = colas, importes, frecuencia, colas_finas
        self.q, self.min_frac_cola = q, min_frac_cola
        self.min_fraudes_importe, self.min_razon_importe = min_fraudes_importe, min_razon_importe
        etiquetas = [n for n, on in (("colas", colas), ("importes", importes), ("frec", frecuencia), ("finas", colas_finas)) if on]
        self.nombre = f"{base.nombre}+{'+'.join(etiquetas) or 'nada'}"

    @staticmethod
    def _importe(col: np.ndarray) -> np.ndarray:
        """Recupera el importe con dos decimales a partir de log1p(importe)."""
        return np.round(np.expm1(col), 2)

    def _extras(self, X: np.ndarray) -> np.ndarray:
        cols = []
        if self.colas:
            tocadas = np.zeros(len(X))
            for j, lado in zip(self._vars, self._lados):
                tocadas += (X[:, j] < self._lo[j]) if lado < 0 else (X[:, j] > self._hi[j])
            cols.append(tocadas)
        if self.colas_finas:
            # Distancia a la cola: 0 para casi todas las filas y valores distintos solo dentro de la cola, de modo que el
            # binning por cuantiles del boosting dedica sus cajas a resolver la cola y no el cuerpo de la distribución.
            for j, lado in zip(self._vars, self._lados):
                cols.append(np.minimum(X[:, j] - self._lo[j], 0.0) if lado < 0 else np.maximum(X[:, j] - self._hi[j], 0.0))
        imp = self._importe(X[:, self.col_importe])
        if self.importes:
            cols.append(np.isin(imp, self._clave).astype(float))
        if self.frecuencia:
            cols.append(np.log1p(np.array([self._conteo.get(v, 0) for v in imp], float)))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AumentoDerivadas":
        if y is None:
            raise ValueError("AumentoDerivadas necesita etiquetas")
        otras = [j for j in range(X.shape[1]) if j != self.col_importe]
        self._lo = np.full(X.shape[1], -np.inf)
        self._hi = np.full(X.shape[1], np.inf)
        self._lo[otras] = np.quantile(X[:, otras], self.q, axis=0)
        self._hi[otras] = np.quantile(X[:, otras], 1 - self.q, axis=0)
        F = X[y == 1]
        self._vars, self._lados = [], []
        for j in otras:
            bajo, alto = (F[:, j] < self._lo[j]).mean(), (F[:, j] > self._hi[j]).mean()
            if max(bajo, alto) >= self.min_frac_cola:
                self._vars.append(j)
                self._lados.append(-1 if bajo >= alto else 1)
        imp = self._importe(X[:, self.col_importe])
        valores, cuentas = np.unique(imp, return_counts=True)
        self._conteo = dict(zip(valores.tolist(), cuentas.tolist()))
        tasa_base = float((y == 1).mean())
        fraudes_por_valor = dict(zip(*np.unique(imp[y == 1], return_counts=True)))
        self._clave = np.array([v for v, f in fraudes_por_valor.items()
                                if f >= self.min_fraudes_importe and f / self._conteo[v] >= self.min_razon_importe * tasa_base])
        self.base.fit(np.column_stack([X, self._extras(X)]), y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.base.score(np.column_stack([X, self._extras(X)]))
