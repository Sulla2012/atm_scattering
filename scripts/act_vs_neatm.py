"""Download ACT (ASTRONAUT) light curves of an asteroid and compare them with NEATM predictions that use a flat
effective emissivity. NEATM is fit to WISE photometry (atm_data sample.db); the model is then evaluated at the
true r, delta, alpha of every ACT observation (JPL Horizons)."""

import argparse
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

from atm.constants import Constants
from atm.functions.emissivity import ghzGridToLambda, ghzListToLambda, lambdaToGHz
from atm.functions.interpolate_flux_lambda import interpFluxLambdaAtObs
from atm.mmfit import checkTableRange, fitEmcee, fluxLambdaToJy, loadWiseFromDb
from atm.models import NEATM
from atm.obs import WISE

AU = Constants.ASTRONOMICAL_UNIT
BUCKET = "s3://cornell-acteroids/"
ARRAYS = ["pa4", "pa5", "pa6"]
FREQS = ["090", "150", "220"]
# Nominal ACT band centres [GHz]; the true bandpass (and spectral-index colour correction) is not modelled
BAND_GHZ = {"090": 98.0, "150": 150.0, "220": 220.0}


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
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--astronaut-dir", type=Path, default=Path("../ASTRONAUT"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/act"))
    ap.add_argument(
        "--table-dir",
        required=True,
        help="NEATM tables covering the ACT bands and the r/alpha range",
    )
    ap.add_argument("--flat-eps", type=float, nargs="+", default=[1.0, 0.9, 0.8])
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--nsteps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("reports/act_vesta"))
    ap.add_argument(
        "--debug", action="store_true", help="one array/band, 10 draws, short chain"
    )
    a = ap.parse_args()
    if a.debug:
        a.n_draws, a.nsteps = 10, 500
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
        nsteps=a.nsteps,
        burnIn=a.nsteps // 3,
        verbose=False,
    )
    post = samples.sample(a.n_draws, random_state=0)
    print(samples.describe().loc[["mean", "std"]].round(4), samples.attrs)

    # --- NEATM at every ACT observation
    tabModel = NEATM(tableDir=a.table_dir)
    lams = {f: float(ghzListToLambda(np.array([g]))[0]) for f, g in BAND_GHZ.items()}
    gridLam = ghzGridToLambda(50, 400, 25)
    tabModel.loadLambdaTables(
        np.unique(np.concatenate([gridLam, list(lams.values())])), verbose=False
    )
    rows, res = [], {}
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
        ref = get_ast_weighted_flux(a.asteroid, arr, f)[
            0
        ]  # ASTRONAUT's weighted flux, scaled to 1 AU / 1 AU / alpha = 0
        mine = weightedMean(
            df["flux"].values * df["weight"].values,
            df["err"].values * df["weight"].values,
        )[0]
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
                "ASTRONAUT_unit_geometry_mJy": float(ref),
                "check_mine_mJy": mine,
            }
        )
    table = pd.DataFrame(rows)
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
    html = (
        f"<h1>{a.asteroid}: ACT vs NEATM (flat emissivity)</h1>"
        f"<p>NEATM fit to WISE: logD = {samples['logD'].median():.3f}, logT1 = {samples['logT1'].median():.3f}, "
        f"chi2 = {samples.attrs['chi2_best']:.1f} / {samples.attrs['ndata']}. Band centres are nominal; no colour correction.</p>"
        f'<img src="act_vs_neatm_sed.png"><img src="act_vs_neatm_scatter.png">{table.round(3).to_html(index=False)}'
    )
    (a.out / "act_vs_neatm.html").write_text(html)
    print(a.out / "act_vs_neatm.html")


if __name__ == "__main__":
    main()
