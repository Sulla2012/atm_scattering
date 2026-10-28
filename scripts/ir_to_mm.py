"""Fit IR photometry (WISE or SPHEREx) with NEATM and predict the mm spectrum with Meteorites-TeraFlash e(f)."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atm.config import Config
from atm.constants import Constants
from atm.functions.emissivity import EmissivityTable, ghzToLambda
from atm.mmfit import _modelFlux, fitEmcee, fluxLambdaToJy, loadWiseFromDb, predictMm
from atm.models import NEATM
from atm.obs import WISE, SPHEREx

c = Constants.LIGHTSPEED
OBS = {"WISE": WISE, "SPHEREx": SPHEREx}


def syntheticData(
    model,
    obs,
    truth: np.ndarray,
    rel_err: float,
    seed: int,
    geometry: tuple[float, float, float],
) -> pd.DataFrame:
    """
    Make three epochs of fake photometry from known (logD, logT1, eps).

    Parameters
    ----------
    model : `~atm.models.NEATM`
    obs : `~atm.obs.Observatory`
    truth : `~numpy.ndarray` (3)
        logD, logT1, eps.
    rel_err : float
        Fractional flux error.
    seed : int
        Random seed.
    geometry : tuple of float
        r [AU], delta [AU], alpha [rad] of the first epoch; later epochs are offset slightly.

    Returns
    -------
    `~pandas.DataFrame`
        Data in the layout expected by `~atm.mmfit.fitEmcee` with columns named by `~atm.Config.columnMapping`
        (WISE) or flux_s01.. (SPHEREx).
    """
    r, delta, alpha = (g + np.array([0.0, 0.01, 0.02]) for g in geometry)
    G = 0.15
    model.loadLambdaTables(obs.filterQuadratureLambdas, verbose=False)
    Y = _modelFlux(model, obs, truth, r, delta, alpha, G)
    err = rel_err * Y
    Y = Y + err * np.random.default_rng(seed).standard_normal(Y.shape)
    df = pd.DataFrame(
        {
            "designation": "synthetic",
            "r_au": r,
            "delta_au": delta,
            "alpha_rad": alpha,
            "G": G,
        }
    )
    for i, name in enumerate(obs.filterNames):
        df[f"flux_{name}_si"], df[f"fluxErr_{name}_si"] = Y[:, i], err[:, i]
    return df


def loadActPoints(
    cache: Path, asteroid: str, geometry: tuple[float, float, float]
) -> list[dict]:
    """
    Stacked ACT flux per array and band, scaled to a given geometry with the paper's Eq. 1.

    Each ASTRONAUT light-curve point gives F_0 = flux * weight (flux at 1 AU / 1 AU / alpha = 0); the
    inverse-variance mean of F_0 is scaled by W = d_earth^-2 d_sun^-1/2 10^(-0.004 alpha[deg]). The paper's
    systematic gain uncertainty (1.4% at f090/f150, 1.8% at f220) is added in quadrature.

    Parameters
    ----------
    cache : `~pathlib.Path`
        Directory holding (or receiving) the ASTRONAUT FITS files.
    asteroid : str
        Name in the ASTRONAUT database.
    geometry : tuple of float
        r [AU], delta [AU], alpha [rad] to scale to.

    Returns
    -------
    list of dict
        Entries with array, GHz (nominal band centre), F_mJy and err_mJy.
    """
    from act_vs_neatm import ARRAYS, BAND_GHZ, FREQS, fetchAct, weightedMean

    r, delta, alpha = geometry
    W = delta**-2 * r**-0.5 * 10 ** (-0.004 * np.degrees(alpha))
    sysErr = {"090": 0.014, "150": 0.014, "220": 0.018}
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for arr in ARRAYS:
        for f in FREQS:
            df = fetchAct(asteroid, arr, f, cache)
            if df is None:
                continue
            F0, err = weightedMean(
                df["flux"].values * df["weight"].values,
                df["err"].values * df["weight"].values,
            )
            out.append(
                {
                    "array": arr,
                    "GHz": BAND_GHZ[f],
                    "F_mJy": F0 * W,
                    "err_mJy": float(np.hypot(err, sysErr[f] * F0)) * W,
                }
            )
    return out


def main() -> None:
    """Parse the CLI, run the fit, predict the mm spectrum and write figures plus an HTML report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", choices=list(OBS), default="WISE")
    ap.add_argument(
        "--data", type=Path, default=None, help="CSV of real data; default: synthetic"
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="atm_data sample.db (WISE only); needs --designation",
    )
    ap.add_argument(
        "--designation",
        default=None,
        help='object designation in the db, e.g. "4" for Vesta',
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.10,
        help="magnitude error added to db photometry",
    )
    ap.add_argument(
        "--model",
        choices=["kirchhoff", "neowise"],
        default="kirchhoff",
        help="kirchhoff: p = (1 - eps) / Q with a fitted eps (see --eps-fixed); neowise: eps fixed at "
        "--neowise-eps in all bands, albedo fitted in W1/W2 only (paper1 run5a)",
    )
    ap.add_argument("--neowise-eps", type=float, default=0.9)
    ap.add_argument(
        "--eps-fixed",
        nargs="+",
        default=[],
        metavar="BAND=EPS",
        help="fixed IR emissivity per band, e.g. W3=0.8 W4=0.98 (paper1 run4a); other bands share the fitted eps",
    )
    ap.add_argument(
        "--emissivity",
        type=Path,
        default=Path("../Meteorites-TeraFlash/results/emissivity/L5_l5c.npz"),
    )
    ap.add_argument(
        "--mm-table-dir",
        default=None,
        help="directory with mm NEATM tables (scripts/build_mm_tables.py)",
    )
    ap.add_argument(
        "--fmin", type=float, default=50.0, help="GHz; must match the built tables"
    )
    ap.add_argument("--fmax", type=float, default=1500.0, help="GHz")
    ap.add_argument(
        "--df", type=float, default=145.0, help="GHz; must match the built tables"
    )
    ap.add_argument(
        "--truth",
        type=float,
        nargs=3,
        default=[3.0, np.log10(300 * np.sqrt(2)), 0.9],
        help="logD logT1 eps",
    )
    ap.add_argument(
        "--geometry",
        type=float,
        nargs=3,
        default=[2.0, 1.2, 0.4],
        help="r [AU] delta [AU] alpha [rad]",
    )
    ap.add_argument("--rel-err", type=float, default=0.05)
    ap.add_argument("--nsteps", type=int, default=3000)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument(
        "--act-cache",
        type=Path,
        default=None,
        help="directory of ASTRONAUT FITS files (downloaded there if missing): overlays the stacked ACT points on the SED",
    )
    ap.add_argument(
        "--act-asteroid", default="Vesta", help="name in the ASTRONAUT database"
    )
    ap.add_argument("--debug", action="store_true", help="short chain, 20 draws")
    a = ap.parse_args()
    if a.debug:
        a.nsteps, a.n_draws = 500, 20
    a.out.mkdir(exist_ok=True)

    obs = OBS[a.obs]()
    model = NEATM()
    cm = dict(Config.columnMapping)
    names = obs.filterNames
    cm["flux_si"], cm["fluxErr_si"] = (
        [f"flux_{n}_si" for n in names],
        [f"fluxErr_{n}_si" for n in names],
    )
    data = (
        loadWiseFromDb(a.db, a.designation, a.sigma, obs)
        if a.db
        else pd.read_csv(a.data)
        if a.data
        else syntheticData(
            model, obs, np.array(a.truth), a.rel_err, 0, tuple(a.geometry)
        )
    )
    epsFixed = {k: float(v) for k, v in (x.split("=") for x in a.eps_fixed)}
    neowise = (
        {"eps": a.neowise_eps, "bands": ["W1", "W2"]} if a.model == "neowise" else None
    )
    samples = fitEmcee(
        model,
        obs,
        data,
        columnMapping=cm,
        epsFixed=epsFixed,
        neowise=neowise,
        nsteps=a.nsteps,
        burnIn=a.nsteps // 3,
        verbose=False,
    )
    print(samples.describe().loc[["mean", "std"]], samples.attrs)

    # mm prediction at the nominal geometry of the first epoch
    f_all = np.arange(a.fmin, a.fmax + a.df / 2, a.df)
    fe = np.load(a.emissivity)["f_ghz"]
    f = f_all[(f_all >= fe[0]) & (f_all <= fe[-1])]
    mmModel = NEATM(
        tableDir=os.path.join(a.mm_table_dir, "")
    )  # trailing slash: the model globs dir + pattern if a.mm_table_dir else model
    r, delta, alpha = (
        float(data[cm[k]].iloc[0]) for k in ["r_au", "delta_au", "alpha_rad"]
    )
    pred = predictMm(
        mmModel, samples, a.emissivity, f, r, delta, alpha, nDraws=a.n_draws
    )
    const = predictMm(
        mmModel,
        samples,
        EmissivityTable(fe, np.ones_like(fe)),
        f,
        r,
        delta,
        alpha,
        nDraws=a.n_draws,
    )

    act = (
        loadActPoints(a.act_cache, a.act_asteroid, (r, delta, alpha))
        if a.act_cache
        else []
    )
    if act:
        fAct = np.array(sorted({p["GHz"] for p in act}))
        constAct = predictMm(
            mmModel,
            samples,
            EmissivityTable(fe, np.ones_like(fe)),
            fAct,
            r,
            delta,
            alpha,
            nDraws=a.n_draws,
        )["med"]
        for (
            pnt
        ) in act:  # model at eps = 1 and the implied flat emissivity at each ACT point
            pnt["model"] = float(constAct[list(fAct).index(pnt["GHz"])])

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    lam = obs.filterEffectiveLambdas
    logD, logT1, third = samples[
        list(samples.columns[:3])
    ].median()  # third: eps, or p for neowise
    ax[0].errorbar(
        lam * 1e6,
        fluxLambdaToJy(data[cm["flux_si"]].values[0], lam),
        fluxLambdaToJy(data[cm["fluxErr_si"]].values[0], lam),
        fmt="o",
    )
    best = _modelFlux(
        model,
        obs,
        np.array([logD, logT1, third]),
        *(data[cm[k]].values.astype(float) for k in ["r_au", "delta_au", "alpha_rad"]),
        float(data[cm["G"]].iloc[0]),
        epsFixed,
        neowise,
    )
    ax[0].plot(lam * 1e6, fluxLambdaToJy(best[0], lam), "-")
    ax[0].set(
        xlabel="um", ylabel="Jy", title=f"IR fit ({a.obs}), epoch 1", yscale="log"
    )
    ax[1].hist2d(samples["logD"], samples["logT1"], bins=40)
    ax[1].set(xlabel="logD", ylabel="logT1", title="posterior")
    ax[2].fill_between(f, pred["lo"], pred["hi"], alpha=0.3)
    ax[2].plot(f, pred["med"], label="e(f)")
    ax[2].plot(f, const["med"], "k--", label="eps = 1")
    ax[2].set(
        xlabel="GHz", ylabel="F_nu [Jy]", title="predicted mm spectrum", yscale="log"
    )
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(a.out / f"ir_to_mm_{a.obs}.png", dpi=100)
    sed, (top, mid, bot) = plt.subplots(
        3, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.3]}
    )
    lam_m = ghzToLambda(f)

    def nuF(F: np.ndarray) -> np.ndarray:
        """nu F_nu [W m^-2] from F_nu [Jy] at the frequencies f [GHz]."""
        return F * f * 1e9 * 1e-26

    def lam4F(F: np.ndarray) -> np.ndarray:
        """lambda^4 F_lambda [W m] from F_nu [Jy] at the frequencies f [GHz] (flat in the Rayleigh-Jeans limit)."""
        return F * 1e-26 * c * lam_m**2

    for ax_, g, label in [
        (top, nuF, r"$\nu F_\nu$ [W m$^{-2}$]"),
        (mid, lam4F, r"$\lambda^4 F_\lambda$ [W m]"),
    ]:
        ax_.fill_between(f, g(pred["lo"]), g(pred["hi"]), alpha=0.3)
        ax_.plot(f, g(pred["med"]), label="NEATM with e(f)")
        ax_.plot(f, g(const["med"]), "k--", label="NEATM with eps = 1")
        ax_.set(xscale="log", yscale="log", ylabel=label)
    markers = {"pa4": "s", "pa5": "o", "pa6": "^"}
    for pnt in act:
        nu, lm = pnt["GHz"] * 1e9, c / (pnt["GHz"] * 1e9)
        y, ye = pnt["F_mJy"] * 1e-29, pnt["err_mJy"] * 1e-29  # W m^-2 Hz^-1
        for ax_, scale in [(top, nu), (mid, c * lm**2)]:
            ax_.errorbar(
                pnt["GHz"],
                y * scale,
                ye * scale,
                fmt=markers[pnt["array"]],
                c="r",
                ms=5,
            )
        bot.errorbar(
            pnt["GHz"],
            pnt["F_mJy"] / (1e3 * pnt["model"]),
            pnt["err_mJy"] / (1e3 * pnt["model"]),
            fmt=markers[pnt["array"]],
            c="r",
            ms=5,
            label=f"ACT {pnt['array']}"
            if pnt["array"] not in [q["array"] for q in act[: act.index(pnt)]]
            else None,
        )
    top.legend()
    ratio = pred["F_nu"] / const["F_nu"]
    bot.fill_between(f, *np.percentile(ratio, [16, 84], axis=0), alpha=0.3)
    bot.plot(f, np.median(ratio, axis=0))
    bot.axhline(1, c="k", ls="--")
    bot.set(xlabel="GHz", ylabel="ratio to eps = 1")
    for ax_ in (top, mid, bot):
        ax_.relim()
        ax_.margins(y=0.12)
        ax_.autoscale_view()
    bot.set_xlim(f[0] * 0.85, f[-1] * 1.2)
    if act:
        bot.legend(fontsize=7, ncol=3)
    sed.tight_layout()
    sed.savefig(a.out / f"ir_to_mm_{a.obs}_sed.png", dpi=100)
    html = (
        f"<h2>IR to mm: {a.obs}</h2><p>emissivity: {a.emissivity.name}; acceptance {samples.attrs['acceptance']:.2f}; "
        f"best chi2 = {samples.attrs['chi2_best']:.1f} for {samples.attrs['ndata']} points.</p>"
        f'<img src="ir_to_mm_{a.obs}.png"><img src="ir_to_mm_{a.obs}_sed.png"><pre>{samples.describe().loc[["mean", "std"]].round(4)}</pre>'
    )
    (a.out / f"ir_to_mm_{a.obs}.html").write_text(html)
    print(a.out / f"ir_to_mm_{a.obs}.html")


if __name__ == "__main__":
    main()
