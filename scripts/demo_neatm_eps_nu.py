"""Compare NEATM spectra with constant vs. frequency-dependent (Meteorites-TeraFlash) emissivity; writes an HTML report."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atm.constants import Constants
from atm.functions.emissivity import EmissivityTable, ghzToLambda
from atm.functions.flux_lambda import calcFluxLambdaAtObs
from atm.models import NEATM

c = Constants.LIGHTSPEED

CAVEATS = """<ul>
<li>e(f) is a hemispherical volume emissivity (two-stream, isothermal half-space, no interface term, zero porosity).</li>
<li>No temperature or emission-angle dependence: lab data at room temperature, normal incidence.</li>
<li>Grey samples (Pallasite, CV3 center/right): below the turnover (~49 GHz) e(f) depends on an assumed grain size.</li>
<li>CV3 thicknesses are provisional. e(f) is only valid inside each sample's fitted band; evaluation outside raises.</li>
<li>Do not use albedoSpecification="auto" (Kirchhoff p=(1-eps)/Q) with a mm emissivity.</li>
</ul>"""


def spectrum(
    model: NEATM,
    table: EmissivityTable | float,
    f_ghz: np.ndarray,
    T_ss: float,
    alpha: float,
    D: float,
) -> np.ndarray:
    """
    NEATM flux density at 1 AU from the observer.

    Parameters
    ----------
    model : `~atm.models.NEATM`
    table : `~atm.functions.EmissivityTable` or float
        Emissivity (callable on wavelength, or constant).
    f_ghz : `~numpy.ndarray` (N)
        Frequencies in GHz.
    T_ss, alpha, D : float
        Subsolar temperature [K], phase angle [rad], diameter [m].

    Returns
    -------
    `~numpy.ndarray` (N)
        F_nu in Jy.
    """
    lam = ghzToLambda(f_ghz)
    F = calcFluxLambdaAtObs(
        model,
        1.0,
        1.0,
        lam,
        np.full_like(lam, T_ss),
        D,
        np.full_like(lam, alpha),
        table,
        threads=1,
    )
    return F * lam**2 / c * 1e26


def main() -> None:
    """Parse the CLI, make figures and write the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emissivity-dir",
        type=Path,
        default=Path("../Meteorites-TeraFlash/results/emissivity"),
    )
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--const-eps", type=float, default=0.9)
    ap.add_argument("--T-ss", type=float, default=300.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--diameter", type=float, default=1e3, help="m")
    ap.add_argument("--n-freq", type=int, default=12)
    ap.add_argument("--debug", action="store_true", help="3 frequencies, 20 draws")
    a = ap.parse_args()
    if a.debug:
        a.n_freq = 3
    a.out.mkdir(exist_ok=True)
    model = NEATM()
    imgs = []
    for path in sorted(a.emissivity_dir.glob("*.npz")):
        d = np.load(path)
        table = EmissivityTable.fromNpz(path)
        f = np.linspace(d["f_ghz"][0], d["f_ghz"][-1], a.n_freq)
        draws = d["e_draws"][: 20 if a.debug else None]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].fill_between(d["f_ghz"], d["e_lo"], d["e_hi"], alpha=0.3)
        ax[0].plot(d["f_ghz"], d["e"])
        ax[0].axhline(a.const_eps, ls="--", c="k")
        ax[0].set(xlabel="GHz", ylabel="effective emissivity", title=str(d["model_id"]))
        ax[1].plot(
            f, spectrum(model, table, f, a.T_ss, a.alpha, a.diameter), label="e(f)"
        )
        ax[1].plot(
            f,
            spectrum(model, a.const_eps, f, a.T_ss, a.alpha, a.diameter),
            "k--",
            label=f"eps={a.const_eps}",
        )
        ax[1].set(xlabel="GHz", ylabel="F_nu [Jy]", title="NEATM", yscale="log")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(a.out / f"{path.stem}.png", dpi=100)
        plt.close(fig)
        imgs.append(
            f'<h2>{path.stem}</h2><p>{len(draws)} draws; fitted band {d["f_ghz"][0]:.0f}-{d["f_ghz"][-1]:.0f} GHz</p><img src="{path.stem}.png">'
        )
    html = f"<html><body><h1>Frequency-dependent emissivity in NEATM</h1>{CAVEATS}{''.join(imgs)}</body></html>"
    (a.out / "emissivity_report.html").write_text(html)
    print(a.out / "emissivity_report.html")


if __name__ == "__main__":
    main()
