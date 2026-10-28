"""Download ACT (ASTRONAUT) light curves of an asteroid and compare them with NEATM predictions that use a flat
effective emissivity. NEATM is fit to WISE photometry (atm_data sample.db); the model is then evaluated at the
true r, delta, alpha of every ACT observation (JPL Horizons)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from atm.config import Config
from atm.constants import Constants
from atm.functions.hg import calcHG, calcQ
from atm.functions.flux_lambda import calcFluxLambdaSun
from atm.functions.emissivity import (
    EmissivityTable,
    ghzGridToLambda,
    ghzListToLambda,
    lambdaToGHz,
)
from atm.functions.interpolate_flux_lambda import interpFluxLambdaAtObs
from atm.mmfit import (
    checkTableRange,
    fitEmcee,
    fitPrescriptions,
    fluxLambdaToJy,
    loadWiseFromDb,
    thermalFluxJy,
)
from atm.models import NEATM
from atm.obs import WISE

AU = Constants.ASTRONOMICAL_UNIT
C = Constants.LIGHTSPEED
BUCKET = "s3://cornell-acteroids/"
ARRAYS = ["pa4", "pa5", "pa6"]
FREQS = ["090", "150", "220"]
# Nominal ACT band centres [GHz], as used in the paper notebooks (lambda = c / nu); the true bandpass and the
# spectral-index colour correction are not modelled
BAND_GHZ = {"090": 90.0, "150": 150.0, "220": 220.0}


def fetchAct(asteroid: str, array: str, freq: str, cache: Path) -> pd.DataFrame | None:
    """
    Get one ASTRONAUT light curve, reading it from the public S3 bucket exactly as `ast_ftns.get_ast_weighted_flux`
    does (astropy + anonymous fsspec) and caching the FITS table locally.

    Parameters
    ----------
    asteroid : str
        Name of the minor object in the database (e.g. "Vesta").
    array, freq : str
        ACT array (pa4, pa5, pa6) and frequency (090, 150, 220).
    cache : `~pathlib.Path`
        Directory holding downloaded FITS files.

    Returns
    -------
    `~pandas.DataFrame` or None
        Columns time (unix), flux and err [mJy], weight; None if the combination is not in the database.
    """
    path = cache / f"{asteroid}_lc_{array}_{freq}.fits"
    if not path.exists():
        try:
            with fits.open(
                f"{BUCKET}{path.stem}.fits", fsspec_kwargs={"anon": True}
            ) as hdul:
                fits.BinTableHDU(hdul[1].data).writeto(path)
        except (PermissionError, FileNotFoundError, OSError):
            print(f"  not in database: {path.stem}")
            return None
    with fits.open(path) as hdul:
        d = hdul[1].data
        return pd.DataFrame(
            {
                "time": d["Time"],
                "flux": d["Flux"],
                "err": d["FluxUncertainty"],
                "weight": d["Weight"],
            }
        )


def getEphemerides(jd: np.ndarray, horizonsId: str, cache: Path) -> pd.DataFrame:
    """
    Geocentric r, delta, alpha at the given Julian dates from JPL Horizons (cached to CSV).

    Parameters
    ----------
    jd : `~numpy.ndarray`
        Julian dates rounded to 5 decimals.
    horizonsId : str
        Horizons small-body id (e.g. "4").
    cache : `~pathlib.Path`
        CSV cache with columns jd, r, delta, alpha_deg.

    Returns
    -------
    `~pandas.DataFrame`
        Ephemerides for all jd, indexed by jd.
    """
    have = (
        pd.read_csv(cache)
        if cache.exists()
        else pd.DataFrame(columns=["jd", "r", "delta", "alpha_deg"])
    )
    need = np.setdiff1d(jd, have["jd"].values)
    rows = [have]
    for i in range(0, len(need), 40):
        for attempt in range(4):
            try:
                e = Horizons(
                    id=horizonsId,
                    id_type="smallbody",
                    location="500",
                    epochs=list(need[i : i + 40]),
                ).ephemerides(quantities="19,20,24")
                break
            except Exception as ex:  # JPL returns transient 5xx errors
                print(f"  Horizons retry: {str(ex)[:60]}")
                time.sleep(5)
        else:
            raise RuntimeError("Horizons query failed")
        rows.append(
            pd.DataFrame(
                {
                    "jd": np.round(np.array(e["datetime_jd"]), 5),
                    "r": e["r"],
                    "delta": e["delta"],
                    "alpha_deg": e["alpha"],
                }
            )
        )
    eph = pd.concat(rows).drop_duplicates("jd")
    eph.to_csv(cache, index=False)
    return eph.set_index("jd").loc[jd]


def modelMjy(
    model: NEATM, lam: float, post: pd.DataFrame, r, delta, alpha
) -> np.ndarray:
    """
    NEATM flux density (eps = 1) of each posterior draw at each observation.

    Parameters
    ----------
    model : `~atm.models.NEATM`
        Model with the table at lam loaded.
    lam : float
        Wavelength in m.
    post : `~pandas.DataFrame`
        Posterior draws with columns logD, logT1.
    r, delta, alpha : `~numpy.ndarray` (N)
        Geometry of the N observations (AU, AU, rad).

    Returns
    -------
    `~numpy.ndarray` (K, N)
        Flux density in mJy for K draws.
    """
    out = np.empty((len(post), len(r)))
    for k, (logD, logT1) in enumerate(zip(post["logD"], post["logT1"])):
        T = 10**logT1 / np.sqrt(r)
        checkTableRange(model, lam, T, alpha)
        out[k] = 1e3 * fluxLambdaToJy(
            interpFluxLambdaAtObs(model, r, delta, lam, T, 10**logD, alpha, 1.0), lam
        )
    return out


def weightedMean(x: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Inverse-variance weighted mean and its uncertainty."""
    w = 1 / err**2
    return float((x * w).sum() / w.sum()), float(w.sum() ** -0.5)


def sedFigure(
    model: NEATM,
    obs: WISE,
    wise: pd.DataFrame,
    samples: pd.DataFrame,
    epsFixed: dict,
    stack: list,
    a,
    alma: list | None = None,
    presc: list | None = None,
) -> None:
    """
    Figure-3-style SED (Orlowski-Scherer et al. 2024): lambda^4 F_lambda of the NEATM model fit to the WISE
    bands, with the ACT stacked fluxes (inverse-variance combined F_0 of every observation, scaled with
    F_i = F_0 W_i, W_i = d_earth^-2 d_sun^-1/2 10^(-0.004 alpha), to the geometry of the WISE observations).

    The model emissivity is interpolated between the WISE bands (fitted eps for W1/W2, then the fixed W3 and W4
    values) and held at the W4 value beyond it; dashed curves instead hold a flat eps beyond W4.

    Parameters
    ----------
    model : `~atm.models.NEATM`
        Model used for direct integration (no tables).
    obs : `~atm.obs.WISE`
        Observatory.
    wise : `~pandas.DataFrame`
        WISE observations used in the fit (with alpha_rad).
    samples : `~pandas.DataFrame`
        Posterior samples from `~atm.mmfit.fitEmcee`.
    epsFixed : dict
        Fixed per-band emissivities used in the fit.
    stack : list of dict
        ACT stacked points with keys array, GHz, F0 [mJy at 1 AU / 1 AU / alpha = 0] and err.
    a : `~argparse.Namespace`
        Parsed CLI arguments (out, flat_eps, n_sed_draws, threads, sys_err).
    """
    cm = Config.columnMapping
    r0, d0, al0 = (float(wise[cm[k]].mean()) for k in ["r_au", "delta_au", "alpha_rad"])
    G = float(wise[cm["G"]].iloc[0])
    lam = np.logspace(-6, -2, 60)
    post = samples.sample(a.n_sed_draws, random_state=1)
    T = 10 ** post["logT1"].values / np.sqrt(r0)
    th = model.calcTotalFluxLambdaEmittedToObsMany(
        np.tile(lam, len(T)),
        np.repeat(T, len(lam)),
        np.full(len(T) * len(lam), al0),
        threads=a.threads,
    ).reshape(len(T), len(lam))
    neowise = a.model == "neowise"
    if neowise:
        eps4 = np.full((len(T), len(obs.filterNames)), a.neowise_eps)
    else:
        eps4 = np.array(
            [
                [
                    post["eps"].values[k] if n not in epsFixed else epsFixed[n]
                    for n in obs.filterNames
                ]
                for k in range(len(T))
            ]
        )
    effL = obs.filterEffectiveLambdas
    edge = 18.5e-6 if neowise else effL[-1]  # start of the 'beyond W4' emissivity
    baseTail = a.neowise_eps if neowise else float(eps4[:, -1].mean())

    def flux(k: int, tail: float | EmissivityTable | None) -> np.ndarray:
        """
        lambda^4 F_lambda [W m] for draw k. Beyond W4 the emissivity is `tail`: None keeps the W4 value, a float
        is a flat eps, and an `~atm.functions.EmissivityTable` is used where it is defined (the W4 value
        in the gap below its range, NaN above its longest wavelength, i.e. no extrapolation, and NaN inside the WISE
        range so that only the part beyond W4 is drawn).
        """
        eps = np.interp(lam, effL, eps4[k])
        nanMask = np.zeros(lam.shape, bool)
        if isinstance(tail, EmissivityTable):
            f = lambdaToGHz(lam)
            inr = (lam > edge) & (f >= tail.f_ghz[0]) & (f <= tail.f_ghz[-1])
            eps = np.where(lam > edge, eps4[k, -1], eps)
            eps[inr] = tail(lam[inr]) * (
                eps4[k, -1] if a.l5_mode == "multiplicative" else 1.0
            )
            nanMask = (lam <= edge) | (f < tail.f_ghz[0])
        elif tail is not None:
            eps = np.where(lam > edge, tail, eps)
        # NEOWISE: albedo p only in W1/W2 (lambda < 6.5 um), none in W3/W4; otherwise Kirchhoff
        p = (
            np.where(lam < 6.5e-6, post["p"].values[k], 0.0)
            if neowise
            else (1 - eps) / calcQ(G)
        )
        D = 10 ** post["logD"].values[k]
        F = (
            D**2
            / (4 * d0**2 * AU**2)
            * (eps * th[k] + p * calcHG(al0, G) * calcFluxLambdaSun(lam, r0))
        )
        return np.where(nanMask, np.nan, F * lam**4)

    rng = np.random.default_rng(2)
    tables = []
    if a.emissivity:
        nEps = len(np.load(a.emissivity)["e_draws"])
        tables = [
            EmissivityTable.fromNpz(a.emissivity, draw=int(i))
            for i in rng.integers(nEps, size=len(T))
        ]
    l5 = np.array([flux(k, tables[k]) for k in range(len(T))]) if tables else None
    fig, ax = plt.subplots(figsize=(8, 5.5))
    base = np.array([flux(k, None) for k in range(len(T))])
    lo, med, hi = np.percentile(base, [2.5, 50, 97.5], axis=0)
    # Spread of the prediction under different emissivity prescriptions (fits of the same data); the formal
    # posterior of one fit does not include the uncertainty of eps(lambda). Evaluated beyond W4 only.
    envB = envL = None
    if presc:
        fits = [
            {
                "name": "this fit",
                "logD": float(post["logD"].median()),
                "logT1": float(post["logT1"].median()),
                "eps_tail": baseTail,
            }
        ] + presc
        tail = lam > edge
        lt = lam[tail]
        thP = np.array(
            [
                model.calcTotalFluxLambdaEmittedToObsMany(
                    lt,
                    np.full(len(lt), 10 ** q["logT1"] / np.sqrt(r0)),
                    np.full(len(lt), al0),
                    threads=a.threads,
                )
                for q in fits
            ]
        )
        scale = (
            np.array([10 ** (2 * q["logD"]) / (4 * d0**2 * AU**2) for q in fits])[
                :, None
            ]
            * thP
            * lt**4
        )
        epsB = np.array([q["eps_tail"] for q in fits])[:, None]
        envB = (np.full(lam.shape, np.nan), np.full(lam.shape, np.nan))
        envB[0][tail], envB[1][tail] = (epsB * scale).min(0), (epsB * scale).max(0)
        if a.emissivity:
            tb = EmissivityTable.fromNpz(a.emissivity, draw="median")
            f_t = lambdaToGHz(lt)
            e_t = np.where(
                (f_t >= tb.f_ghz[0]) & (f_t <= tb.f_ghz[-1]),
                tb(np.clip(lt, C / (tb.f_ghz[-1] * 1e9), C / (tb.f_ghz[0] * 1e9))),
                np.nan,
            )
            epsL = epsB if a.l5_mode == "multiplicative" else 1.0
            envL = (np.full(lam.shape, np.nan), np.full(lam.shape, np.nan))
            envL[0][tail], envL[1][tail] = (
                (e_t * epsL * scale).min(0),
                (e_t * epsL * scale).max(0),
            )
        for g in [90.0, 150.0, 220.0]:
            j = np.argmin(np.abs(lt - C / (g * 1e9)))
            v = (epsB * scale)[:, j]  # row 0 is this fit
            print(
                f"emissivity-prescription spread of the baseline prediction at {g:.0f} GHz: "
                f"{100 * (v.min() / v[0] - 1):+.1f}% / {100 * (v.max() / v[0] - 1):+.1f}% relative to this fit"
            )
    ax.fill_between(lam * 1e6, lo, hi, alpha=0.25, color="C0")
    ax.plot(
        lam * 1e6,
        med,
        c="C0",
        label=("NEOWISE NEATM fit to WISE" if neowise else "NEATM fit to WISE")
        + f" (eps beyond W4 = {baseTail:.2f}), 95%",
    )
    flats = [x for x in a.flat_eps if abs(x - baseTail) > 1e-9]
    for i, x in enumerate(flats, start=1):
        ax.plot(
            lam * 1e6,
            np.median([flux(k, x) for k in range(len(T))], axis=0),
            ls="--",
            c=f"C{i}",
            label=f"flat eps = {x:g} beyond W4",
        )
    if l5 is not None:
        l5lo, l5med, l5hi = np.nanpercentile(l5, [2.5, 50, 97.5], axis=0)
        ax.fill_between(lam * 1e6, l5lo, l5hi, alpha=0.25, color="C4")
        ax.plot(
            lam * 1e6,
            l5med,
            c="C4",
            lw=2,
            label=("eps_IR x L5 e(f)" if a.l5_mode == "multiplicative" else "L5 e(f)")
            + f" beyond W4 ({Path(a.emissivity).stem}), 95%",
        )
    if envB is not None:
        ax.fill_between(
            lam * 1e6,
            *envB,
            facecolor="none",
            edgecolor="0.35",
            hatch="///",
            lw=0,
            label="emissivity-prescription spread",
        )
    if envL is not None:
        ax.fill_between(
            lam * 1e6,
            *envL,
            facecolor="none",
            edgecolor="C4",
            hatch="\\\\",
            lw=0,
            label="same, with L5 e(f)",
        )
    for j, n in enumerate(
        obs.filterNames
    ):  # median flux and median error per band, as plotObservations(plotMedian=True)
        ax.errorbar(
            effL[j] * 1e6,
            np.median(wise[cm["flux_si"][j]]) * effL[j] ** 4,
            np.median(wise[cm["fluxErr_si"][j]]) * effL[j] ** 4,
            fmt="o",
            c="k",
            ms=3,
            label="WISE (median)" if j == 0 else None,
        )
    Wgeo = d0**-2 * r0**-0.5 * 10 ** (-0.004 * np.degrees(al0))
    sysErr = {90.0: 0.014, 150.0: 0.014, 220.0: 0.018}
    markers = {"pa4": "s", "pa5": "o", "pa6": "^"}
    zoom, labelled = [], set()
    for pnt in stack:
        nu = pnt["GHz"] * 1e9
        lm = 299792458.0 / nu
        y = pnt["F0"] * Wgeo * 1e-29 * 299792458.0 * lm**2
        ye = (
            np.hypot(pnt["err"], sysErr[pnt["GHz"]] * pnt["F0"])
            * Wgeo
            * 1e-29
            * 299792458.0
            * lm**2
        )
        ax.errorbar(
            lm * 1e6,
            y,
            ye,
            fmt=markers[pnt["array"]],
            c="r",
            ms=5,
            label=f"ACT {pnt['array']}" if pnt["array"] not in labelled else None,
        )
        labelled.add(pnt["array"])
        zoom.append((lm * 1e6, y, ye, markers[pnt["array"]]))
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Wavelength [$\mu$m]",
        ylabel=r"$\lambda^4 F_\lambda$ [W m]",
        xlim=(0.7, 2e4),
    )
    ax.set_title(
        f"{a.asteroid}: r={r0:.2f} AU, delta={d0:.2f} AU, alpha={np.degrees(al0):.0f} deg (WISE geometry)",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ins = ax.inset_axes([0.45, 0.08, 0.52, 0.4])
    ins.fill_between(lam * 1e6, lo, hi, alpha=0.25, color="C0")
    if envB is not None:
        ins.fill_between(
            lam * 1e6, *envB, facecolor="none", edgecolor="0.35", hatch="///", lw=0
        )
    if envL is not None:
        ins.fill_between(
            lam * 1e6, *envL, facecolor="none", edgecolor="C4", hatch="\\\\", lw=0
        )
    ins.plot(lam * 1e6, med, c="C0")
    for i, x in enumerate(flats, start=1):
        ins.plot(
            lam * 1e6,
            np.median([flux(k, x) for k in range(len(T))], axis=0),
            ls="--",
            c=f"C{i}",
        )
    if l5 is not None:
        ins.fill_between(lam * 1e6, l5lo, l5hi, alpha=0.25, color="C4")
        ins.plot(lam * 1e6, l5med, c="C4", lw=2)
    for pnt in alma or []:
        lm = 299792458.0 / (pnt["GHz"] * 1e9)
        Fa = thermalFluxJy(
            model, lm, post, (pnt["r_au"], pnt["delta_au"], pnt["alpha_rad"]), a.threads
        )
        Fr = thermalFluxJy(model, lm, post, (r0, d0, al0), a.threads)
        scale = float(np.median(Fr / Fa))  # NEATM-based scaling to the WISE geometry
        y, ye = (
            pnt[k] * 1e-3 * scale * 1e-26 * 299792458.0 * lm**2
            for k in ("F_mJy", "err_mJy")
        )
        ax.errorbar(lm * 1e6, y, ye, fmt="D", c="g", ms=6, label=pnt["label"])
        zoom.append((lm * 1e6, y, ye, "D"))
        ins.errorbar(lm * 1e6, y, ye, fmt="D", c="g", ms=5)
        eps = pnt["F_mJy"] * 1e-3 / float(np.median(Fa))
        eL5 = (
            float(
                np.interp(
                    pnt["GHz"], *(np.load(a.emissivity)[k] for k in ("f_ghz", "e"))
                )
            )
            if a.emissivity
            else np.nan
        )
        print(
            f"{pnt['label']}: implied eps = {eps:.3f} +/- {pnt['err_mJy'] * 1e-3 / float(np.median(Fa)):.3f}; L5 eps_mm = {eL5 * (baseTail if a.l5_mode == 'multiplicative' else 1.0):.3f}; implied / L5 = {eps / (eL5 * (baseTail if a.l5_mode == 'multiplicative' else 1.0)):.3f}"
        )
    if alma:  # refresh the legend so that the ALMA entry appears
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for x, y, ye, m in zoom:
        if m != "D":
            ins.errorbar(x, y, ye, fmt=m, c="r", ms=4)
    ins.set(xlim=(1.1e3, 3.7e3))
    ymin = min(y - ye for _, y, ye, _ in zoom)
    ins.set_ylim(0.8 * ymin, 1.2 * hi[(lam * 1e6 > 1.1e3) & (lam * 1e6 < 3.7e3)].max())
    ins.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(a.out / "act_sed_lambda4.png", dpi=110, bbox_inches="tight")


def main() -> None:
    """Parse the CLI, run the WISE fit, compare with ACT and write the figures and report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--asteroid", default="Vesta", help="name in the ASTRONAUT database"
    )
    ap.add_argument("--horizons-id", default="4")
    ap.add_argument("--db", type=Path, default=Path("../atm_data/paper1/sample.db"))
    ap.add_argument("--designation", default="4", help="designation in the WISE db")
    ap.add_argument(
        "--eps-fixed", nargs="+", default=["W3=0.8", "W4=0.98"], metavar="BAND=EPS"
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        help="mag error offset (paper notebook class: 0.15)",
    )
    ap.add_argument(
        "--model",
        choices=["neowise", "kirchhoff"],
        default="neowise",
        help="neowise: eps fixed, albedo free in W1/W2 (paper1 run5a); kirchhoff: p = (1 - eps) / Q with --eps-fixed",
    )
    ap.add_argument("--neowise-eps", type=float, default=0.9)
    ap.add_argument("--astronaut-dir", type=Path, default=Path("../ASTRONAUT"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/act"))
    ap.add_argument(
        "--table-dir",
        required=True,
        help="NEATM tables covering the ACT bands and the r/alpha range",
    )
    ap.add_argument("--flat-eps", type=float, nargs="+", default=[1.0, 0.9, 0.8])
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument(
        "--emissivity",
        type=lambda x: Path(x) if x else None,
        default=Path("../Meteorites-TeraFlash/results/emissivity/L5_l5c.npz"),
        help="TeraFlash e(f) NPZ overlaid on the SED (pass an empty string to skip)",
    )
    ap.add_argument(
        "--alma-json",
        type=Path,
        default=None,
        help="JSON from scripts/alma_flux.py, overlaid on the lambda^4 F_lambda SED",
    )
    ap.add_argument(
        "--l5-mode",
        choices=["multiplicative", "absolute"],
        default="multiplicative",
        help="multiplicative: eps_mm(nu) = eps_IR x e(nu) (e(nu) distorts the IR emissivity continued beyond W4); "
        "absolute: eps_mm(nu) = e(nu)",
    )
    ap.add_argument(
        "--no-systematics",
        action="store_true",
        help="skip the emissivity-prescription fits (envelope on the SED)",
    )
    ap.add_argument(
        "--n-sed-draws",
        type=int,
        default=30,
        help="posterior draws for the direct-integration SED band",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("reports/act_vesta"))
    ap.add_argument(
        "--debug", action="store_true", help="one array/band, 10 draws, short chain"
    )
    a = ap.parse_args()
    if a.debug:
        a.n_draws, a.nsteps, a.n_sed_draws = 10, 500, 3
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.astronaut_dir))
    from ast_ftns import (
        get_ast_weighted_flux,
    )  # ASTRONAUT: used as a cross-check of the weighted flux

    # --- ACT data
    groups = (
        [("pa5", "150")] if a.debug else [(arr, f) for arr in ARRAYS for f in FREQS]
    )
    act = {}
    for arr, f in groups:
        df = fetchAct(a.asteroid, arr, f, a.cache_dir)
        if df is not None:
            act[(arr, f)] = df
    print("ACT data:", {k: len(v) for k, v in act.items()})
    jd = {
        k: np.round(Time(v["time"].values, format="unix").jd, 5) for k, v in act.items()
    }
    eph = getEphemerides(
        np.unique(np.concatenate(list(jd.values()))),
        a.horizons_id,
        a.cache_dir / "ephemerides.csv",
    )

    # --- IR fit
    obs, model = WISE(), NEATM()
    wise = loadWiseFromDb(a.db, a.designation, a.sigma, obs)
    epsFixed = {k: float(v) for k, v in (x.split("=") for x in a.eps_fixed)}
    samples = fitEmcee(
        model,
        obs,
        wise,
        epsFixed=epsFixed,
        neowise={"eps": a.neowise_eps, "bands": ["W1", "W2"]}
        if a.model == "neowise"
        else None,
        nsteps=a.nsteps,
        burnIn=a.nsteps // 3,
        verbose=False,
    )
    post = samples.sample(a.n_draws, random_state=0)
    # IR emissivity continued beyond W4 (distorted by e(nu) in the multiplicative mode)
    epsTail = (
        a.neowise_eps
        if a.model == "neowise"
        else epsFixed["W4"]
        if "W4" in epsFixed
        else float(samples["eps"].median())
    )
    print(samples.describe().loc[["mean", "std"]].round(4), samples.attrs)

    # --- NEATM at every ACT observation
    tabModel = NEATM(
        tableDir=os.path.join(a.table_dir, "")
    )  # trailing slash: the model globs dir + pattern
    lams = {f: float(ghzListToLambda(np.array([g]))[0]) for f, g in BAND_GHZ.items()}
    gridLam = ghzGridToLambda(50, 400, 25)
    tabModel.loadLambdaTables(
        np.unique(np.concatenate([gridLam, list(lams.values())])), verbose=False
    )
    rows, res, stack = [], {}, []
    if a.emissivity:
        l5 = np.load(a.emissivity)
        l5f, l5e = l5["f_ghz"], l5["e"]
    for (arr, f), df in act.items():
        e = eph.loc[jd[(arr, f)]]
        r, delta, alpha = (
            e["r"].values,
            e["delta"].values,
            np.radians(e["alpha_deg"].values),
        )
        M = modelMjy(tabModel, lams[f], post, r, delta, alpha)  # (K, N), eps = 1
        Mmed = np.median(M, axis=0)
        eps_k = np.array(
            [weightedMean(df["flux"].values / m, df["err"].values / m)[0] for m in M]
        )
        eps_best, eps_stat = weightedMean(
            df["flux"].values / Mmed, df["err"].values / Mmed
        )
        chi2 = {
            x: float((((df["flux"].values - x * Mmed) / df["err"].values) ** 2).sum())
            for x in a.flat_eps + [eps_best]
        }
        eL5 = float(np.interp(BAND_GHZ[f], l5f, l5e)) if a.emissivity else np.nan
        if a.l5_mode == "multiplicative":
            eL5 *= epsTail  # eps_mm = eps_IR x e(nu)
        L5 = (
            eL5 * Mmed
        )  # NEATM (eps = 1) x the L5-distorted emissivity at the band centre
        ratioL5, ratioL5Err = (
            weightedMean(df["flux"].values / L5, df["err"].values / L5)
            if a.emissivity
            else (np.nan, np.nan)
        )
        ref = get_ast_weighted_flux(a.asteroid, arr, f)[
            0
        ]  # ASTRONAUT's weighted flux, scaled to 1 AU / 1 AU / alpha = 0
        mine, mineErr = weightedMean(
            df["flux"].values * df["weight"].values,
            df["err"].values * df["weight"].values,
        )
        stack.append({"array": arr, "GHz": BAND_GHZ[f], "F0": mine, "err": mineErr})
        res[(arr, f)] = dict(
            df=df, M=M, Mmed=Mmed, geom=(r, delta, alpha), eps_best=eps_best
        )
        rows.append(
            {
                "array": arr,
                "GHz": BAND_GHZ[f],
                "N": len(df),
                "eps_implied": eps_best,
                "stat_err": eps_stat,
                "NEATM_draw_spread": float(eps_k.std()),
                **{f"chi2_eps{x:g}": chi2[x] for x in a.flat_eps},
                "chi2_best": chi2[eps_best],
                "eps_mm_L5": eL5,
                "chi2_L5": float(
                    (((df["flux"].values - L5) / df["err"].values) ** 2).sum()
                ),
                "ratio_to_L5": ratioL5,
                "ratio_to_L5_err": ratioL5Err,
                "ASTRONAUT_unit_geometry_mJy": float(ref),
                "check_mine_mJy": mine,
            }
        )
    table = pd.DataFrame(rows)
    comb = pd.DataFrame(
        [
            {
                "GHz": g,
                "eps_combined": np.average(
                    d["eps_implied"], weights=1 / d["stat_err"] ** 2
                ),
                "stat_err": float((1 / d["stat_err"] ** 2).sum() ** -0.5),
            }
            for g, d in table.groupby("GHz")
        ]
    )
    print(comb.round(3).to_string(index=False))
    pd.set_option("display.width", 250)
    print(table.round(3).to_string())

    # --- figures
    n = len(res)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, ((arr, f), d) in zip(axs[0], res.items()):
        df, Mm = d["df"], d["Mmed"]
        ax.errorbar(
            Mm, df["flux"], df["err"], fmt="o", ms=3, c="k", alpha=0.6, label="ACT"
        )
        xs = np.linspace(0, Mm.max() * 1.05, 2)
        for x in a.flat_eps:
            ax.plot(xs, x * xs, label=f"flat eps={x:g}")
        ax.plot(xs, d["eps_best"] * xs, "k--", label=f"best {d['eps_best']:.2f}")
        if a.emissivity:
            ax.plot(
                xs,
                np.interp(BAND_GHZ[f], l5f, l5e)
                * (epsTail if a.l5_mode == "multiplicative" else 1.0)
                * xs,
                "m-",
                lw=2,
                label="L5",
            )
        ax.set(
            xlabel="NEATM (eps=1) [mJy]",
            ylabel="ACT [mJy]",
            title=f"{arr} {BAND_GHZ[f]:g} GHz",
        )
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / "act_vs_neatm_scatter.png", dpi=100)

    # reference geometry: median over all observations
    allE = eph
    rr, dd, aa = (
        allE["r"].median(),
        allE["delta"].median(),
        np.radians(allE["alpha_deg"].median()),
    )
    curveF = ghzGridToLambda(50, 400, 25)
    fGHz = lambdaToGHz(curveF)
    ref = np.array(
        [
            modelMjy(
                tabModel, lc, post, np.array([rr]), np.array([dd]), np.array([aa])
            )[:, 0]
            for lc in curveF
        ]
    ).T  # (K, F)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lo, med, hi = np.percentile(ref, [16, 50, 84], axis=0)
    for x in a.flat_eps:
        ax[0].plot(fGHz, x * med, label=f"NEATM flat eps={x:g}")
    ax[0].fill_between(
        fGHz,
        a.flat_eps[0] * lo,
        a.flat_eps[0] * hi,
        alpha=0.25,
        label="WISE-fit uncertainty (eps=1)",
    )
    for (arr, f), d in res.items():
        df = d["df"]
        Mref = np.median(
            modelMjy(
                tabModel, lams[f], post, np.array([rr]), np.array([dd]), np.array([aa])
            )[:, 0]
        )
        scaled, serr = weightedMean(
            df["flux"].values * Mref / d["Mmed"], df["err"].values * Mref / d["Mmed"]
        )
        ax[0].errorbar(BAND_GHZ[f], scaled, serr, fmt="o", c="k")
    ax[0].plot([], [], "ko", label="ACT (scaled to reference geometry)")
    ax[0].set(
        xlabel="GHz",
        ylabel="F_nu [mJy]",
        yscale="log",
        xscale="log",
        title=f"r={rr:.2f} AU, delta={dd:.2f} AU, alpha={np.degrees(aa):.0f} deg",
    )
    ax[0].legend(fontsize=7)
    ax[1].errorbar(
        table["GHz"],
        table["eps_implied"],
        table["stat_err"],
        fmt="o",
        c="k",
        label="ACT / NEATM(eps=1)",
    )
    for x in a.flat_eps:
        ax[1].axhline(x, ls=":", c="gray")
    ax[1].set(
        xlabel="GHz",
        ylabel="implied flat effective emissivity",
        title="weighted ACT / NEATM ratio",
    )
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(a.out / "act_vs_neatm_sed.png", dpi=100)
    alma = json.loads(a.alma_json.read_text()) if a.alma_json else []
    presc = (
        None
        if a.no_systematics
        else fitPrescriptions(model, obs, wise, Config.columnMapping)
    )
    if presc:
        print(
            "prescription fits:",
            [
                (q["name"], round(10 ** q["logD"] / 1e3, 1), round(q["chi2"]))
                for q in presc
            ],
        )
    sedFigure(model, obs, wise, samples, epsFixed, stack, a, alma, presc)
    html = (
        f"<h1>{a.asteroid}: ACT vs NEATM (flat emissivity)</h1>"
        f"<p>NEATM fit to WISE: logD = {samples['logD'].median():.3f}, logT1 = {samples['logT1'].median():.3f}, "
        f"chi2 = {samples.attrs['chi2_best']:.1f} / {samples.attrs['ndata']}. Band centres are nominal; no colour correction.</p>"
        f'<img src="act_sed_lambda4.png"><img src="act_vs_neatm_sed.png"><img src="act_vs_neatm_scatter.png">{comb.round(3).to_html(index=False)}{table.round(3).to_html(index=False)}'
    )
    (a.out / "act_vs_neatm.html").write_text(html)
    print(a.out / "act_vs_neatm.html")


if __name__ == "__main__":
    main()
