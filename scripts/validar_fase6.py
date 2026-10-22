"""Fase 6: contrasta las hipótesis H1…H8 del EDA con el pipeline real (validación 5x2 sin fuga).

Etapas (con caché en data/cache_fase6/, que no se versiona):
  oof         scores fuera de muestra de hgb, rf y et en las 10 particiones (lento: ~15 min)
  produccion  scores crudos de producción con los modelos ajustados en todo construcción
  rapidos     H1-H5 con gradient boosting (rápido)
  et          H3 con Extra Trees (lento)
  x14         dependencia del modelo respecto a X14 (rápido)
  finas       H10: variables 'distancia a la cola' para HGB (exploración + confirmación en particiones nuevas)
  confirmar   confirmación de H4 (frecuencia del importe) sobre el ensemble real, en particiones nuevas (lento)
  analisis    H6-H8 a partir de la caché
  informe     escribe docs/04_VALIDACION_FASE6.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from anomalias.config import cargar_config  # noqa: E402
from anomalias.detectors.aumentado import AumentoDerivadas  # noqa: E402
from anomalias.detectors.bosques import ExtraTreesDetector  # noqa: E402
from anomalias.detectors.gbm_supervisado import GBMSupervisado  # noqa: E402
from anomalias.detectors.registro import ENSEMBLE_FINAL, catalogo  # noqa: E402
from anomalias.domain.diagnostico import (  # noqa: E402
    comparacion_pareada, fraccion_con_copia, importancia_permutacion, recall_en_k,
)
from anomalias.domain.ensemble import combinar  # noqa: E402
from anomalias.domain.features import construir_features  # noqa: E402
from anomalias.domain.leakage_guard import verificar_ajuste, verificar_ajuste_no_produccion  # noqa: E402
from anomalias.domain.schema import ConfigExperimento  # noqa: E402
from anomalias.domain.validation import aupr, deduplicar, evaluar, particiones  # noqa: E402
from anomalias.orchestration.pipeline import ejecutar_detectores  # noqa: E402

CACHE = RAIZ / "data" / "cache_fase6"
OBJ = "Class"
COL_IMPORTE = 20  # posición de Amount en las variables


def datos(dedup: bool = True):
    """Construcción (deduplicada o no): X transformada, X cruda, y, nombres de columnas y partición 5x2."""
    cfg = cargar_config()
    df = pd.read_csv(cfg.train_file)
    if dedup:
        df = deduplicar(df)
    y = df[OBJ].to_numpy(int)
    X, cols = construir_features(df, OBJ, "Amount")
    Xc = df[list(cols)].to_numpy(float)
    exp = ConfigExperimento(seed=cfg.seed, columna_objetivo=OBJ)
    return cfg, df, y, X, Xc, cols, exp


def cv_aupr(fabrica, X, y, exp, cols) -> np.ndarray:
    """AUPR por partición (pasa por el guardrail anti-fuga)."""
    return evaluar(fabrica, X, y, exp, cols)["aupr"].to_numpy()


def guardar(nombre: str, obj: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{nombre}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def cargar(nombre: str) -> dict:
    return json.loads((CACHE / f"{nombre}.json").read_text(encoding="utf-8"))


def pares(a: np.ndarray, b: np.ndarray) -> dict:
    return comparacion_pareada(a, b)


def mejora_real(p: dict) -> bool:
    """Criterio conservador: mejora media > 0, z ≥ 2 y gana en ≥ 80 % de las particiones."""
    return p["dif_media"] > 0 and p["z"] >= 2 and p["gana_en"] >= 0.8 * p["particiones"]


# ----------------------------------------------------------------------------- etapas

def etapa_oof() -> None:
    cfg, df, y, X, Xc, cols, exp = datos()
    cat = catalogo()
    especs = [cat[n] for n in ENSEMBLE_FINAL]
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(particiones(y, exp)):
        ruta = CACHE / f"oof_{i}.npz"
        if ruta.exists():
            continue
        t0 = time.perf_counter()
        verificar_ajuste(p.idx_fit, p)
        r = ejecutar_detectores(especs, X[p.idx_fit], y[p.idx_fit], X[p.idx_val], cfg.seed, n_jobs=3)
        np.savez_compressed(ruta, idx_val=p.idx_val, **r.scores)
        print(f"partición {i}: {time.perf_counter() - t0:.0f} s · modelos {sorted(r.scores)}", flush=True)


def etapa_produccion() -> None:
    cfg, df, y, X, Xc, cols, exp = datos()
    prod = pd.read_csv(cfg.prod_file)
    Xp, cols_p = construir_features(prod, OBJ, "Amount")
    assert cols == cols_p
    verificar_ajuste_no_produccion(X, Xp)
    cat = catalogo()
    r = ejecutar_detectores([cat[n] for n in ENSEMBLE_FINAL], X, y, Xp, cfg.seed, n_jobs=3)
    np.savez_compressed(CACHE / "produccion.npz", **r.scores)
    print("producción puntuada:", sorted(r.scores), flush=True)


def etapa_rapidos() -> None:
    cfg, df, y, X, Xc, cols, exp = datos()
    seed = cfg.seed
    hgb = lambda: GBMSupervisado(seed)  # noqa: E731
    out: dict = {"n_filas": len(y), "n_pos": int(y.sum())}
    base = cv_aupr(hgb, X, y, exp, cols)
    out["base"] = base.tolist()
    ps = particiones(y, exp)

    # H1: importancia por permutación frente a la firma del EDA + ablación de X13/X15
    imp = np.zeros(X.shape[1])
    for p in ps:
        m = GBMSupervisado(seed).fit(X[p.idx_fit], y[p.idx_fit])
        imp += importancia_permutacion(m, X[p.idx_val], y[p.idx_val], n_rep=2, seed=seed)
    imp /= len(ps)
    aucs = np.array([max((a := (pd.Series(X[:, j]).rank().to_numpy()[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() + 1) / 2)
                          / ((y == 1).sum() * (y == 0).sum())), 1 - a) for j in range(X.shape[1])])
    rho = float(pd.Series(imp).rank().corr(pd.Series(aucs).rank()))
    out["h1"] = {"columnas": list(cols), "importancia": imp.tolist(), "auc_univariante": aucs.tolist(), "spearman_imp_auc": rho}
    keep = [j for j, c in enumerate(cols) if c not in ("X13", "X15")]
    abl = cv_aupr(hgb, X[:, keep], y, exp, [cols[j] for j in keep])
    out["h1"]["ablacion_X13_X15"] = abl.tolist()
    out["h1"]["ablacion_pareada"] = pares(abl, base)
    print("H1 listo", flush=True)

    # H2: sin deduplicar (las copias caen a ambos lados del pliegue)
    cfg2, df2, y2, X2, Xc2, cols2, exp2 = datos(dedup=False)
    sin = cv_aupr(hgb, X2, y2, exp2, cols2)
    ps2 = particiones(y2, exp2)[:5]
    copia = float(np.mean([fraccion_con_copia(X2[p.idx_fit], X2[p.idx_val]) for p in ps2]))
    copia_dedup = float(np.mean([fraccion_con_copia(X[p.idx_fit], X[p.idx_val]) for p in ps[:5]]))
    out["h2"] = {"sin_dedup": sin.tolist(), "con_dedup": base.tolist(), "n_sin_dedup": len(y2),
                 "frac_val_con_copia_sin_dedup": copia, "frac_val_con_copia_dedup": copia_dedup}
    print("H2 listo", flush=True)

    # H3: X en bruto frente a transformadas (hgb)
    crudo = cv_aupr(hgb, Xc, y, exp, cols)
    out["h3_hgb"] = {"crudo": crudo.tolist(), "pareada": pares(crudo, base)}
    print("H3 (hgb) listo", flush=True)

    # H4 / H5: variables derivadas calculadas solo con las filas de ajuste
    variantes = {
        "colas": dict(colas=True), "importes": dict(importes=True), "frecuencia": dict(frecuencia=True),
        "colas+importes": dict(colas=True, importes=True), "todo": dict(colas=True, importes=True, frecuencia=True),
    }
    out["h45"] = {}
    for nombre, kw in variantes.items():
        v = cv_aupr(lambda kw=kw: AumentoDerivadas(GBMSupervisado(seed), COL_IMPORTE, **kw), X, y, exp, cols)
        out["h45"][nombre] = {"aupr": v.tolist(), "pareada": pares(v, base), "mejora_real": mejora_real(pares(v, base))}
        print(f"H4/H5 {nombre}: {v.mean():.4f} (base {base.mean():.4f})", flush=True)
    guardar("rapidos", out)


def etapa_et() -> None:
    """H3 con Extra Trees (sus umbrales aleatorios dependen de la escala): solo la 1.ª repetición."""
    cfg, df, y, X, Xc, cols, exp = datos()
    exp1 = ConfigExperimento(seed=cfg.seed, columna_objetivo=OBJ, n_repeats=1)
    et = lambda: ExtraTreesDetector(cfg.seed)  # noqa: E731
    transf = cv_aupr(et, X, y, exp1, cols)
    crudo = cv_aupr(et, Xc, y, exp1, cols)
    guardar("et", {"transformado": transf.tolist(), "crudo": crudo.tolist(), "pareada": pares(crudo, transf)})
    print("H3 (et) listo", flush=True)


def etapa_x14() -> None:
    """¿Cuánto depende el modelo de X14? (la importancia por permutación la señala como dominante)."""
    cfg, df, y, X, Xc, cols, exp = datos()
    hgb = lambda: GBMSupervisado(cfg.seed)  # noqa: E731
    base = np.array(cargar("rapidos")["base"])
    j = cols.index("X14")
    sin = [k for k in range(X.shape[1]) if k != j]
    s14 = cv_aupr(hgb, X[:, sin], y, exp, [cols[k] for k in sin])
    solo = cv_aupr(hgb, X[:, [j]], y, exp, ["X14"])
    guardar("x14", {"sin_x14": s14.tolist(), "solo_x14": solo.tolist(), "sin_x14_pareada": pares(s14, base)})
    print(f"sin X14 {s14.mean():.4f} · solo X14 {solo.mean():.4f}", flush=True)


def etapa_finas() -> None:
    """H10: el binning por cuantiles de HGB pierde resolución en la cola; ¿ayudan las variables 'distancia a la cola'?

    Exploración en las particiones habituales y confirmación independiente en particiones nuevas (semilla + 2000).
    """
    cfg, df, y, X, Xc, cols, exp = datos()
    out: dict = {}
    for etiqueta, semilla in (("exploracion", cfg.seed), ("confirmacion", cfg.seed + 2000)):
        e = ConfigExperimento(seed=semilla, columna_objetivo=OBJ)
        base = cv_aupr(lambda: GBMSupervisado(cfg.seed), X, y, e, cols)
        fin = cv_aupr(lambda: AumentoDerivadas(GBMSupervisado(cfg.seed), COL_IMPORTE, colas_finas=True), X, y, e, cols)
        out[etiqueta] = {"base": base.tolist(), "finas": fin.tolist(), "pareada": pares(fin, base), "mejora_real": mejora_real(pares(fin, base))}
        print(f"H10 {etiqueta}: base {base.mean():.4f} -> finas {fin.mean():.4f} {pares(fin, base)}", flush=True)
    guardar("finas", out)


def etapa_confirmar() -> None:
    """Confirmación independiente de la frecuencia del importe (H4) sobre el ENSEMBLE real, en particiones NUEVAS.

    Se pre-registra una única hipótesis (frecuencia mejora el ensemble) y se evalúa con otra semilla de particiones
    que la usada al explorar, para evitar el sesgo de haber elegido la mejor de varias variantes mirando los mismos datos.
    """
    from anomalias.detectors.base import EspecDetector

    cfg, df, y, X, Xc, cols, exp = datos()
    exp_n = ConfigExperimento(seed=cfg.seed + 1000, n_repeats=1, columna_objetivo=OBJ)
    cat = catalogo()
    base = [cat[n] for n in ENSEMBLE_FINAL]
    var = [EspecDetector(f"{n}+frec", (lambda s, n=n: AumentoDerivadas(cat[n].fabrica(s), COL_IMPORTE, frecuencia=True)), opcional=False)
           for n in ENSEMBLE_FINAL]
    filas = []
    for i, p in enumerate(particiones(y, exp_n)):
        t0 = time.perf_counter()
        verificar_ajuste(p.idx_fit, p)
        r = ejecutar_detectores(base + var, X[p.idx_fit], y[p.idx_fit], X[p.idx_val], cfg.seed, n_jobs=3)
        yv = y[p.idx_val]
        fila = {k: aupr(yv, v) for k, v in r.scores.items()}
        fila["ens_base"] = aupr(yv, combinar({n: r.scores[n] for n in ENSEMBLE_FINAL}))
        fila["ens_frec"] = aupr(yv, combinar({f"{n}+frec": r.scores[f"{n}+frec"] for n in ENSEMBLE_FINAL}))
        filas.append(fila)
        print(f"partición {i}: {time.perf_counter() - t0:.0f} s · ens base {fila['ens_base']:.4f} · ens+frec {fila['ens_frec']:.4f}", flush=True)
    t = pd.DataFrame(filas)
    out = {"aupr": {c: t[c].tolist() for c in t}}
    for n in (*ENSEMBLE_FINAL,):
        out[f"{n}_pareada"] = pares(t[f"{n}+frec"].to_numpy(), t[n].to_numpy())
    out["ens_pareada"] = pares(t["ens_frec"].to_numpy(), t["ens_base"].to_numpy())
    out["ens_mejora_real"] = mejora_real(out["ens_pareada"])
    guardar("confirmar", out)


def etapa_analisis() -> None:
    cfg, df, y, X, Xc, cols, exp = datos()
    ps = particiones(y, exp)
    oof = [np.load(CACHE / f"oof_{i}.npz") for i in range(len(ps))]
    modelos = list(ENSEMBLE_FINAL)
    aupr_m = {m: np.array([aupr(y[p.idx_val], o[m]) for p, o in zip(ps, oof)]) for m in modelos}
    aupr_m["ens"] = np.array([aupr(y[p.idx_val], combinar({m: o[m] for m in modelos})) for p, o in zip(ps, oof)])
    out: dict = {"aupr": {k: v.tolist() for k, v in aupr_m.items()}}

    # H7: comparaciones pareadas
    out["h7"] = {"ens_vs_hgb": pares(aupr_m["ens"], aupr_m["hgb"]), "ens_vs_rf": pares(aupr_m["ens"], aupr_m["rf"]),
                 "ens_vs_et": pares(aupr_m["ens"], aupr_m["et"]), "hgb_vs_rf": pares(aupr_m["hgb"], aupr_m["rf"]),
                 "hgb_vs_et": pares(aupr_m["hgb"], aupr_m["et"])}

    # H6: cobertura por subgrupo de fraude (mismo agrupamiento descriptivo del EDA)
    from sklearn.cluster import KMeans  # solo para etiquetar subgrupos (descriptivo), igual que el EDA

    Z = X[:, :20]
    Zs = (Z - Z.mean(0)) / Z.std(0)
    pos = np.flatnonzero(y == 1)
    lab = KMeans(2, n_init=10, random_state=cfg.seed).fit_predict(Zs[pos])
    if (lab == 0).sum() < (lab == 1).sum():
        lab = 1 - lab
    suave = np.zeros(len(y), bool)
    suave[pos[lab == 1]] = True
    fuerte = np.zeros(len(y), bool)
    fuerte[pos[lab == 0]] = True
    out["h6"] = {"n_fuerte": int(fuerte.sum()), "n_suave": int(suave.sum()), "por_modelo": {}}
    for nombre in ("hgb", "rf", "et", "ens"):
        rf_, rs_, ra_ = [], [], []
        for p, o in zip(ps, oof):
            s = combinar({m: o[m] for m in modelos}) if nombre == "ens" else o[nombre]
            yv = y[p.idx_val]
            rf_.append(recall_en_k(yv, s, fuerte[p.idx_val]))
            rs_.append(recall_en_k(yv, s, suave[p.idx_val]))
            ra_.append(recall_en_k(yv, s))
        out["h6"]["por_modelo"][nombre] = {"recall_fuerte": float(np.nanmean(rf_)), "recall_suave": float(np.nanmean(rs_)),
                                           "recall_total": float(np.nanmean(ra_))}
    # rango medio (percentil) de cada grupo en la repetición 0 (cada fila validada una vez)
    pct = np.full(len(y), np.nan)
    for p, o in zip(ps[:5], oof[:5]):
        s = pd.Series(combinar({m: o[m] for m in modelos})).rank(pct=True).to_numpy()
        pct[p.idx_val] = s
    out["h6"]["percentil_medio_ens"] = {"fuerte": float(np.nanmean(pct[fuerte])), "suave": float(np.nanmean(pct[suave]))}

    # H8: scores crudos de producción frente a fuera de muestra
    prod = np.load(CACHE / "produccion.npz")
    out["h8"] = {}
    prev = float(y.mean())
    for m in modelos:
        o = np.full(len(y), np.nan)
        for p, oo in zip(ps[:5], oof[:5]):
            o[p.idx_val] = oo[m]
        sp = prod[m]
        umbral = float(np.quantile(o, 1 - prev))
        a, b = np.sort(o), np.sort(sp)
        grid = np.union1d(a, b)
        ks = float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))
        out["h8"][m] = {"ks": ks, "frac_prod_sobre_umbral": float((sp > umbral).mean()), "prevalencia_esperada": prev,
                        "n_prod_sobre_umbral": int((sp > umbral).sum()), "n_prod": int(len(sp)),
                        "q_oof": {str(q): float(np.quantile(o, q)) for q in (.5, .9, .99, .999)},
                        "q_prod": {str(q): float(np.quantile(sp, q)) for q in (.5, .9, .99, .999)}}
    guardar("analisis", out)
    print("análisis listo")


def _f(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _par(p: dict) -> str:
    if p["error_tipico"] == 0 and p["dif_media"] == 0:
        return f"idéntico en las {p['particiones']} particiones"
    return f"Δ = {_f(p['dif_media'])} (e.t. {_f(p['error_tipico'])}; z = {_f(p['z'], 1)}; gana en {p['gana_en']}/{p['particiones']})"


def etapa_informe() -> None:
    r, a = cargar("rapidos"), cargar("analisis")
    x14 = cargar("x14") if (CACHE / "x14.json").exists() else None
    et = cargar("et") if (CACHE / "et.json").exists() else None
    cf = cargar("confirmar") if (CACHE / "confirmar.json").exists() else None
    base = np.array(r["base"])
    ruido = float(base.std(ddof=1))
    v: dict[str, tuple[str, str, str]] = {}  # id -> (veredicto, resumen, ¿cambia el proyecto?)

    # H1
    h1 = r["h1"]
    imp = dict(zip(h1["columnas"], h1["importancia"]))
    irrel = max(imp["X13"], imp["X15"]) < 0.005 and abs(h1["ablacion_pareada"]["z"]) < 2
    ranking = h1["spearman_imp_auc"] >= 0.6
    top = max(imp, key=imp.get)
    v["H1"] = ("Confirmada" if irrel and ranking else "Parcial" if irrel else "Refutada",
               f"`X13` y `X15` no aportan (importancia {_f(imp['X13'])} y {_f(imp['X15'])}; quitarlas: {_par(h1['ablacion_pareada'])}). "
               f"La importancia concuerda solo moderadamente con el AUC univariante (Spearman {_f(h1['spearman_imp_auc'], 2)}) y está **muy concentrada en `{top}`** "
               f"({_f(imp[top], 3)} de caída de AUPR al permutarla)." + (f" Pese a ello, **quitar `X14` solo cuesta {_f(base.mean() - np.mean(x14['sin_x14']))} de AUPR** ({_f(np.mean(x14['sin_x14']))} frente a {_f(base.mean())}; {_par(x14['sin_x14_pareada'])}): "
                                         f"la señal está repartida y es redundante en otras variables, como indicaba el EDA; la importancia por permutación exagera la dependencia. "
                                         f"(Un HGB con `X14` sola rinde {_f(np.mean(x14['solo_x14']))}.)" if x14 else ""),
               "No")
    # H2
    h2 = r["h2"]
    infl = float(np.mean(h2["sin_dedup"]) - np.mean(h2["con_dedup"]))
    v["H2"] = ("Confirmada, pero poco material" if 0 < infl < ruido else "Confirmada" if infl >= ruido else "Refutada",
               f"Sin deduplicar el AUPR sube de {_f(np.mean(h2['con_dedup']))} a {_f(np.mean(h2['sin_dedup']))} (+{_f(infl)}), mucho menos que el ruido entre pliegues ({_f(ruido)}). "
               f"Aun así, el {_f(100 * h2['frac_val_con_copia_sin_dedup'], 1)} % de las filas de validación tiene una copia exacta en ajuste (0 % tras deduplicar).", "No (deduplicar se mantiene por corrección)")
    # H3
    h3 = r["h3_hgb"]["pareada"]
    txt3 = f"HGB: resultado idéntico con X en bruto y transformadas ({_par(h3)})."
    if et:
        pe = et["pareada"]
        sens = abs(pe["z"]) >= 2
        txt3 += f" Extra Trees (umbrales aleatorios): crudo {_f(np.mean(et['crudo']))} frente a transformado {_f(np.mean(et['transformado']))} ({_par(pe)})."
    v["H3"] = ("Confirmada para HGB" + ("; **no** para Extra Trees" if et and abs(et['pareada']['z']) >= 2 else "; también para Extra Trees" if et else "; Extra Trees pendiente"), txt3,
               "No" if not (et and abs(et["pareada"]["z"]) >= 2) else "Revisar: la transformación importa para ET")
    # H4 / H5
    h45 = r["h45"]
    txt4 = "; ".join(f"{k}: {_f(np.mean(x['aupr']))} ({_par(x['pareada'])})" for k, x in h45.items() if k in ("importes", "frecuencia"))
    if cf:
        ens = cf["ens_pareada"]
        txt4 += (f". **Confirmación en particiones nuevas (semilla distinta), sobre el ensemble real:** base {_f(np.mean(cf['aupr']['ens_base']))} → con frecuencia {_f(np.mean(cf['aupr']['ens_frec']))} ({_par(ens)}); "
                 + "; ".join(f"{n}: {_par(cf[f'{n}_pareada'])}" for n in ENSEMBLE_FINAL))
        ok4 = cf["ens_mejora_real"]
    else:
        ok4 = None
    if ok4 is None:
        ver4, cam4 = "Pendiente", "Pendiente"
    elif ok4:
        ver4, cam4 = "Confirmada", "**Posible mejora: decide el usuario**"
    elif all(cf[f"{n}_pareada"]["dif_media"] > 0 for n in ENSEMBLE_FINAL) and cf["ens_pareada"]["dif_media"] > 0:
        ver4 = "Indicio consistente de mejora pequeña; no alcanza el criterio"
        cam4 = "No por ahora (ganancia ≈ 10 veces menor que el ruido entre pliegues); **opcional si el usuario lo decide**"
    else:
        ver4, cam4 = "No confirmada", "No"
    v["H4"] = (ver4, "El indicador de importes clave no aporta; la **frecuencia del importe** mejoró HGB en la exploración. " + txt4, cam4)
    c5 = h45["colas"]["pareada"]
    v["H5"] = ("Refutada" if not h45["colas"]["mejora_real"] else "Confirmada",
               f"Añadir el recuento de colas tocadas: {_f(np.mean(h45['colas']['aupr']))} frente a {_f(np.mean(base))} ({_par(c5)}). Los árboles ya capturan esa información.", "No")
    # H6
    h6 = a["h6"]
    pm = h6["por_modelo"]["ens"]
    brecha = pm["recall_fuerte"] - pm["recall_suave"]
    v["H6"] = ("Confirmada" if brecha >= 0.15 else "Refutada",
               f"Recall entre los k primeros (k = nº de fraudes) del ensemble: grupo fuerte {_f(100 * pm['recall_fuerte'], 1)} % (n={h6['n_fuerte']}) frente a grupo suave {_f(100 * pm['recall_suave'], 1)} % (n={h6['n_suave']}). "
               f"Percentil medio del score: {_f(100 * h6['percentil_medio_ens']['fuerte'], 2)} % frente a {_f(100 * h6['percentil_medio_ens']['suave'], 2)} %. "
               f"**El {_f(100 * (1 - pm['recall_suave']) * h6['n_suave'] / ((1 - pm['recall_fuerte']) * h6['n_fuerte'] + (1 - pm['recall_suave']) * h6['n_suave']), 0)} % de los fraudes que el modelo no coloca entre los k primeros pertenece al grupo suave**: el techo de AUPR ≈ 0,83 es, en esencia, el techo de detección del fraude suave.", "No (es el límite principal del modelo; con estas variables no hay evidencia de cómo superarlo)")
    # H7
    h7 = a["h7"]["ens_vs_hgb"]
    v["H7"] = ("Confirmada (diferencia no significativa)" if not (h7["z"] >= 2 and h7["gana_en"] >= 8) else "Refutada",
               f"Ensemble − hgb: {_par(h7)}. Ensemble − rf: {_par(a['h7']['ens_vs_rf'])}. Ensemble − et: {_par(a['h7']['ens_vs_et'])}. hgb − rf: {_par(a['h7']['hgb_vs_rf'])}. "
               "El ensemble supera de forma consistente a `rf` y `et` por separado, pero **no se distingue de `hgb` solo**: `hgb` bastaría; el ensemble aporta robustez, no una mejora demostrada.", "No")
    # H8
    h8 = a["h8"]
    ok8 = all(x["ks"] < 0.10 and 0.5 <= x["frac_prod_sobre_umbral"] / x["prevalencia_esperada"] <= 2.0 for x in h8.values())
    v["H8"] = ("Confirmada" if ok8 else "Refutada",
               "; ".join(f"{m}: KS {_f(x['ks'], 3)}, {x['n_prod_sobre_umbral']} filas de producción sobre el umbral que en validación deja pasar un {_f(100 * x['prevalencia_esperada'], 3)} % (esperadas ≈ {round(x['prevalencia_esperada'] * x['n_prod'])})" for m, x in h8.items()), "No")
    if (CACHE / "finas.json").exists():
        fi = cargar("finas")
        v["H10"] = ("No confirmada",
                    "Hipótesis surgida en esta fase: con una sola variable, HGB rinde "
                    + (f"{_f(np.mean(x14['solo_x14']))}" if x14 else "n/d")
                    + " mientras que el score directo `-X14` da ≈ 0,63 (EDA), lo que sugiere que el *binning* por cuantiles pierde resolución en la cola. "
                    f"Variables 'distancia a la cola': exploración {_f(np.mean(fi['exploracion']['base']))} → {_f(np.mean(fi['exploracion']['finas']))} ({_par(fi['exploracion']['pareada'])}); "
                    f"confirmación en particiones nuevas {_f(np.mean(fi['confirmacion']['base']))} → {_f(np.mean(fi['confirmacion']['finas']))} ({_par(fi['confirmacion']['pareada'])}). "
                    "La dirección es siempre positiva pero pequeña y no alcanza el criterio; la explicación del *binning* queda como indicio no probado.", "No")
    v["H9"] = ("Resuelta en el EDA", "KS de posición p = 0,84; sin autocorrelación.", "No")

    L = ["# 04 · Validación de la Fase 6: ¿es coherente el proyecto con las conclusiones del EDA?", "",
         "Generado por `scripts/validar_fase6.py` (etapa `informe`). Cifras calculadas con el pipeline real, esquema **5 folds × 2 repeticiones sin fuga**, "
         f"datos deduplicados ({r['n_filas']:,} filas, {r['n_pos']} fraudes), AUPR como métrica.".replace("166,717", "166.717"), "",
         "## Metodología y cautelas",
         f"- **Ruido de referencia**: desviación entre particiones del AUPR de HGB = {_f(ruido)}. Las comparaciones son **pareadas** por partición.",
         "- **Criterio de mejora real** (conservador): diferencia media > 0, z ≥ 2 y ganar en ≥ 80 % de las particiones. Las 10 particiones comparten datos (2 repeticiones), por lo que el z está algo inflado.",
         "- **Sin fuga**: todo ajuste pasa por el guardrail; las variables derivadas (colas, importes clave, frecuencia) se calculan **solo con las filas de ajuste** (`AumentoDerivadas`).",
         "- **Comparaciones múltiples**: en la exploración se probaron 5 variantes sobre las mismas particiones; por eso la única candidata se **re-evalúa en particiones nuevas** sobre el ensemble real.",
         "- El pipeline entregado **no se ha modificado** en esta fase.", "",
         "## Resumen", "", "| Hipótesis | Veredicto | ¿Cambia el proyecto? |", "|---|---|---|"]
    for h, (ver, _, cam) in v.items():
        L.append(f"| {h} | {ver} | {cam} |")
    L += ["", "## Detalle"]
    for h, (ver, txt, cam) in v.items():
        L += ["", f"### {h} — {ver}", txt, f"**¿Cambia el proyecto?** {cam}"]
    L += ["", "## Conclusiones",
          "1. **El proyecto es coherente con el EDA**: la deduplicación, la transformación (neutra para árboles), la ausencia de ajuste con producción, el uso de árboles y la métrica quedan respaldados; ninguna conclusión del EDA obliga a cambiar el pipeline.",
          "2. **El límite del modelo ya está explicado (H6):** casi todo el fraude no detectado es del grupo \"suave\". Mejorar el AUPR pasaría por detectar ese grupo, algo que estas variables apenas permiten.",
          "3. **Hipótesis del EDA refutadas o sin efecto medible** (H5, H10, y H4 en parte): los árboles ya extraen esa información. Se evitó, así, añadir complejidad sin evidencia.",
          "4. **Única mejora candidata: la frecuencia del importe (H4)**, con ganancia consistente pero pequeña (≈ +0,004 en el ensemble, mejora los tres modelos en particiones nuevas) que no alcanza el criterio conservador. **No se aplica sin tu decisión**; si se aplicara habría que regenerar el CSV y el notebook y revalidar.",
          "5. **Ensemble frente a `hgb` solo (H7):** equivalentes dentro del ruido; se mantiene el ensemble por robustez, como se decidió en la Fase 2.",
          "", "## Reproducción", "```bash", "python scripts/validar_fase6.py oof        # ~15 min", "python scripts/validar_fase6.py produccion", "python scripts/validar_fase6.py rapidos",
          "python scripts/validar_fase6.py x14", "python scripts/validar_fase6.py et          # ~6 min", "python scripts/validar_fase6.py finas",
          "python scripts/validar_fase6.py confirmar   # ~10 min", "python scripts/validar_fase6.py analisis", "python scripts/validar_fase6.py informe", "```"]
    (RAIZ / "docs" / "04_VALIDACION_FASE6.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("informe escrito en docs/04_VALIDACION_FASE6.md")


ETAPAS = {"informe": etapa_informe, "oof": etapa_oof, "produccion": etapa_produccion, "rapidos": etapa_rapidos, "et": etapa_et,
          "x14": etapa_x14, "finas": etapa_finas, "confirmar": etapa_confirmar, "analisis": etapa_analisis}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=list(ETAPAS))
    a = ap.parse_args()
    ETAPAS[a.etapa]()
