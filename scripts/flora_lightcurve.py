"""Fit the WISE photometry of an asteroid with a rotational lightcurve instead of a flat error floor, and compare the
NEOWISE-style NEATM fits (precision, chi^2, and the leave-one-band-out sensitivity of the mm prediction)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atm.config import Config
from atm.constants import Constants
from atm.mmfit import (
    _modelFlux,
    fitEmcee,
    lightcurveModulation,
    loadWiseFromDb,
    thermalFluxJy,
)
from atm.models import NEATM
from atm.obs import WISE

C = Constants.LIGHTSPEED
NEO = {"eps": 0.9, "bands": ["W1", "W2"]}


def summarize(
    label: str,
    s: pd.DataFrame,
    model: NEATM,
    geometry: tuple[float, float, float],
    ghz: float,
    threads: int,
) -> dict:
    """
    One row of the comparison table.

    Parameters
    ----------
    label : str
        Name of the fit.
    s : `~pandas.DataFrame`
        Posterior samples from `~atm.mmfit.fitEmcee`.
    model : `~atm.models.NEATM`
        Flux model.
    geometry : tuple of float
        r [AU], delta [AU], alpha [rad] at which the thermal flux (eps = 1) is evaluated.
    ghz : float
        Frequency of the prediction.
    threads : int
        Processes for the integrations.

    Returns
    -------
    dict
        D and its width, T1 width, the 150 GHz-style flux (median and 1-sigma %), and chi^2.
    """
    F = (
        thermalFluxJy(
            model, C / (ghz * 1e9), s.sample(60, random_state=0), geometry, threads
        )
        * 1e3
    )
    q = np.percentile(F, [16, 50, 84])
    return {
        "fit": label,
        "D_km": float(np.median(10 ** s["logD"]) / 1e3),
        "sigma_D_pct": float(100 * np.log(10) * s["logD"].std()),
        "sigma_T1_pct": float(100 * np.log(10) * s["logT1"].std()),
        "F_mJy": float(q[1]),
        "sigma_F_pct": float(100 * (q[2] - q[0]) / 2 / q[1]),
        "chi2": float(s.attrs["chi2_best"]),
        "N": int(s.attrs["ndata"]),
    }


def main() -> None:
    """Parse the CLI, run the fits, write the table, figure and HTML report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("../atm_data/paper1/sample.db"))
    ap.add_argument("--designation", default="8")
    ap.add_argument(
        "--period-h",
        type=float,
        default=12.865,
        help="rotation period [h] (LCDB for 8 Flora)",
    )
    ap.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    ap.add_argument("--baseline-floor", type=float, default=0.15)
    ap.add_argument("--ghz", type=float, default=150.0)
    ap.add_argument(
        "--geometry",
        type=float,
        nargs=3,
        default=[2.050, 1.117, float(np.radians(15.2))],
        metavar=("R", "DELTA", "ALPHA_RAD"),
    )
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("reports/flora"))
    ap.add_argument(
        "--debug", action="store_true", help="short chains, one lightcurve floor"
    )
    a = ap.parse_args()
    if a.debug:
        a.nsteps, a.floors = 800, [0.02]
    a.out.mkdir(parents=True, exist_ok=True)
    obs, model = WISE(), NEATM()
    cm = Config.columnMapping
    geom = tuple(a.geometry)
    kw = dict(nsteps=a.nsteps, burnIn=a.nsteps // 3, verbose=False, neowise=NEO)
    lc = {"period_h": a.period_h}

    rows, fits = [], {}
    base = loadWiseFromDb(a.db, a.designation, a.baseline_floor, obs)
    s = fitEmcee(model, obs, base, **kw)
    rows.append(
        summarize(
            f"no lightcurve, floor {a.baseline_floor:.2f} mag",
            s,
            model,
            geom,
            a.ghz,
            a.threads,
        )
    )
    for fl in a.floors:
        w = loadWiseFromDb(a.db, a.designation, fl, obs)
        s = fitEmcee(model, obs, w, lightcurve=lc, **kw)
        fits[fl] = (w, s)
        rows.append(
            summarize(
                f"lightcurve, floor {fl:.2f} mag", s, model, geom, a.ghz, a.threads
            )
        )
    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))

    # leave-one-band-out with the lightcurve fit at the smallest floor that gives a sensible chi^2
    ref = min(
        fits,
        key=lambda f: abs(
            fits[f][1].attrs["chi2_best"] / fits[f][1].attrs["ndata"] - 1
        ),
    )
    w0 = fits[ref][0]
    drop = []
    for i, b in enumerate(obs.filterNames):
        w = w0.copy()
        w[cm["flux_si"][i]] = np.nan
        s = fitEmcee(model, obs, w, lightcurve=lc, **kw)
        drop.append(
            summarize(
                f"lightcurve floor {ref:.2f}, drop {b}",
                s,
                model,
                geom,
                a.ghz,
                a.threads,
            )
        )
    dropTable = pd.DataFrame(drop)
    print(dropTable.round(2).to_string(index=False))

    # lightcurve figure: data / (unmodulated best-fit model) against rotational phase
    w, s = fits[ref]
    best = s.loc[s["logprob"].idxmax()]
    th = best[["logD", "logT1", "p"]].values.astype(float)
    t = w[cm["exp_mjd"]].values
    phase = ((t - t.min()) / (a.period_h / 24)) % 1
    model0 = _modelFlux(
        model,
        obs,
        th,
        w["r_au"].values,
        w["delta_au"].values,
        w["alpha_rad"].values,
        float(w["G"].iloc[0]),
        None,
        NEO,
    )
    pg = np.linspace(0, 1, 200)
    mod = lightcurveModulation(
        t.min() + pg * a.period_h / 24,
        best[["lc_a2", "lc_b2", "lc_a1", "lc_b1"]].values.astype(float),
        a.period_h,
    )
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, bnd in enumerate(obs.filterNames):
        ax[0].errorbar(
            phase,
            w[cm["flux_si"][i]] / model0[:, i],
            w[cm["fluxErr_si"][i]] / model0[:, i],
            fmt="o",
            ms=3,
            label=bnd,
        )
    ax[0].plot(pg, mod, "k-", label="fitted lightcurve")
    ax[0].set(
        xlabel=f"rotational phase (P = {a.period_h} h)",
        ylabel="data / unmodulated model",
        title="WISE lightcurve fit",
    )
    ax[0].legend(fontsize=7, ncol=3)
    ax[1].errorbar(
        range(len(table)),
        table["F_mJy"],
        table["F_mJy"] * table["sigma_F_pct"] / 100,
        fmt="o",
    )
    ax[1].set(
        xticks=range(len(table)),
        xticklabels=[x.replace(", ", "\n") for x in table["fit"]],
        ylabel=f"thermal flux at {a.ghz:g} GHz (eps = 1) [mJy]",
        title="prediction vs treatment of variability",
    )
    ax[1].tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(a.out / "flora_lightcurve.png", dpi=100)

    amp = {k: float(best[k]) for k in ["lc_a2", "lc_b2", "lc_a1", "lc_b1"]}
    (a.out / "flora_lightcurve.json").write_text(
        json.dumps(
            {
                "fits": rows,
                "drop_one_band": drop,
                "best_lightcurve_coeffs": amp,
                "period_h": a.period_h,
            },
            indent=1,
        )
    )
    html = (
        "<h1>Flora: WISE lightcurve fit</h1>"
        f"<p>Rotation period {a.period_h} h (LCDB). The lightcurve is a flux multiplier common to all bands: "
        "1 + a2 cos 4 pi p + b2 sin 4 pi p + a1 cos 2 pi p + b1 sin 2 pi p.</p>"
        f'<img src="flora_lightcurve.png">{table.round(2).to_html(index=False)}'
        f"<h2>Leave one band out (lightcurve fit)</h2>{dropTable.round(2).to_html(index=False)}"
    )
    (a.out / "flora_lightcurve.html").write_text(html)
    print(a.out / "flora_lightcurve.html")


if __name__ == "__main__":
    main()
