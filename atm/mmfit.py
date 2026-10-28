from __future__ import annotations

from pathlib import Path

import sqlite3

import emcee
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Config
from .data_processing import modifyErrors
from .constants import Constants
from .functions.emissivity import EmissivityTable, ghzToLambda
from .functions.hg import calcQ
from .functions.interpolate_flux_lambda import (
    interpFluxLambdaAtObs,
    interpFluxLambdaObsWithSunlight,
)

__all__ = [
    "checkTableRange",
    "fitEmcee",
    "fluxLambdaToJy",
    "PRESCRIPTIONS",
    "fitPrescriptions",
    "loadWiseFromDb",
    "prescriptionFluxJy",
    "thermalFluxJy",
    "predictMm",
]

AU = Constants.ASTRONOMICAL_UNIT
c = Constants.LIGHTSPEED
FIT_PARAMETERS = ["logD", "logT1", "eps"]
T_SS_LIMITS = Config.tableParameterLimits["T_ss"][0]


def fluxLambdaToJy(flux: np.ndarray, lambd: np.ndarray) -> np.ndarray:
    """
    Convert F_lambda to F_nu.

    Parameters
    ----------
    flux : `~numpy.ndarray`
        Flux in W m^-3.
    lambd : float or `~numpy.ndarray`
        Wavelength in m (broadcastable to flux).

    Returns
    -------
    `~numpy.ndarray`
        Flux density in Jy.
    """
    return flux * np.asarray(lambd) ** 2 / c * 1e26


def checkTableRange(model, lambd: float, T_ss, alpha) -> None:
    """
    Raise if T_ss or alpha fall outside the grid of the loaded lookup table at lambd (the spline
    would otherwise silently clamp to the edge).

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model with tables loaded.
    lambd : float
        Wavelength in m (a loaded table key).
    T_ss : float or `~numpy.ndarray`
        Subsolar temperature in K.
    alpha : float or `~numpy.ndarray`
        Phase angle in radians.

    Raises
    ------
    ValueError
        If any value lies outside the table.
    """
    tx, ty = model.loadedLambdaTables[lambd].get_knots()
    for name, v, lo, hi in [
        ("T_ss", T_ss, tx[0], tx[-1]),
        ("alpha", alpha, ty[0], ty[-1]),
    ]:
        if np.min(v) < lo or np.max(v) > hi:
            raise ValueError(
                f"{name} range [{np.min(v):.4g}, {np.max(v):.4g}] outside table [{lo:.4g}, {hi:.4g}]"
            )


def loadWiseFromDb(db, designation: str, sigma: float, obs) -> pd.DataFrame:
    """
    Load clipped WISE observations of one object from an atm_data sample database, following
    the paper1 procedure (keep == 1, then `~atm.modifyErrors` adds sigma magnitudes of error).

    Parameters
    ----------
    db : str or `~pathlib.Path`
        sqlite database with an `observations` table (e.g. atm_data/paper1/sample.db).
    designation : str
        Object designation as stored in the database (e.g. "4" for Vesta).
    sigma : float
        Magnitude error offset added by `~atm.modifyErrors`.
    obs : `~atm.obs.WISE`
        Observatory used to convert the magnitude errors to flux errors.

    Returns
    -------
    `~pandas.DataFrame`
        Observations with an `alpha_rad` column.
    """
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        df = pd.read_sql(
            "SELECT * FROM observations WHERE designation = ? AND keep = 1",
            con,
            params=(designation,),
        )
    if df.empty:
        raise ValueError(f"No kept observations of {designation} in {db}")
    df = modifyErrors(df, obs, sigma=sigma)
    df["alpha_rad"] = np.radians(df["alpha_deg"])
    return df


def thermalFluxJy(
    model,
    lambd: float,
    post: pd.DataFrame,
    geometry: tuple[float, float, float],
    threads: int = 1,
) -> np.ndarray:
    """
    Thermal NEATM flux density (eps = 1, no reflected light) of each posterior draw, by direct integration
    (no lookup tables), at one wavelength and observing geometry.

    Parameters
    ----------
    model : `~atm.models.NEATM`
        Flux model.
    lambd : float
        Wavelength in m.
    post : `~pandas.DataFrame`
        Posterior draws with columns logD, logT1.
    geometry : tuple of float
        r [AU], delta [AU], alpha [rad] of the observation.
    threads : int, optional
        Number of processes for the integrations.

    Returns
    -------
    `~numpy.ndarray` (K)
        Flux density in Jy for K draws.
    """
    r, delta, alpha = geometry
    T = 10 ** post["logT1"].values / np.sqrt(r)
    th = model.calcTotalFluxLambdaEmittedToObsMany(
        np.full(len(T), lambd), T, np.full(len(T), alpha), threads=threads
    )
    return fluxLambdaToJy(
        10 ** (2 * post["logD"].values) / (4 * delta**2 * AU**2) * th, lambd
    )


# Emissivity prescriptions for the WISE fit (Moeyens et al. 2020, Sec. 2.4; paper1 runs): each is a set of keyword
# arguments for `fitEmcee`. The emissivity continued beyond W4 (to the mm) is the fixed W4 value.
PRESCRIPTIONS = {
    "NEOWISE eps=0.8": {"neowise": {"eps": 0.8, "bands": ["W1", "W2"]}},
    "NEOWISE eps=1.0": {"neowise": {"eps": 1.0, "bands": ["W1", "W2"]}},
    "eps_W3,W4=0.80,0.98": {"epsFixed": {"W3": 0.8, "W4": 0.98}},
    "eps_W3,W4=0.76,0.93": {"epsFixed": {"W3": 0.76, "W4": 0.93}},
    "eps_W3,W4=0.70,0.86": {"epsFixed": {"W3": 0.7, "W4": 0.86}},
    "eps_W3,W4=0.90,0.90": {"epsFixed": {"W3": 0.9, "W4": 0.9}},
}


def fitPrescriptions(
    model,
    obs,
    data: pd.DataFrame,
    columnMapping: dict = Config.columnMapping,
    prescriptions: dict | None = None,
    nsteps: int = 1500,
    burnIn: int = 500,
) -> list[dict]:
    """
    Fit the same WISE data under several emissivity prescriptions to measure the systematic spread of the
    thermal prediction (the formal posterior of a single fit does not include the uncertainty of eps(lambda)).

    Parameters
    ----------
    model : `~atm.models.NEATM`
        Flux model.
    obs : `~atm.obs.WISE`
        Observatory.
    data : `~pandas.DataFrame`
        WISE observations (see `loadWiseFromDb`).
    columnMapping : dict, optional
        [Default = `~atm.Config.columnMapping`]
    prescriptions : dict, optional
        Name -> keyword arguments for `fitEmcee` (epsFixed or neowise).
        [Default = `PRESCRIPTIONS`]
    nsteps, burnIn : int, optional
        Chain length per fit.

    Returns
    -------
    list of dict
        One entry per prescription with name, logD, logT1 (posterior medians), eps_tail (the emissivity continued
        beyond W4) and chi2.
    """
    out = []
    for name, kw in (prescriptions or PRESCRIPTIONS).items():
        s = fitEmcee(
            model,
            obs,
            data,
            columnMapping=columnMapping,
            nsteps=nsteps,
            burnIn=burnIn,
            verbose=False,
            **kw,
        )
        tail = kw["neowise"]["eps"] if "neowise" in kw else kw["epsFixed"]["W4"]
        out.append(
            {
                "name": name,
                "logD": float(s["logD"].median()),
                "logT1": float(s["logT1"].median()),
                "eps_tail": float(tail),
                "chi2": s.attrs["chi2_best"],
            }
        )
    return out


def prescriptionFluxJy(
    model,
    lambd: float,
    fit: dict,
    geometry: tuple[float, float, float],
    threads: int = 1,
) -> float:
    """
    Thermal NEATM flux density (eps = 1) at the posterior median of one prescription fit.

    Parameters
    ----------
    model : `~atm.models.NEATM`
        Flux model.
    lambd : float
        Wavelength in m.
    fit : dict
        One entry from `fitPrescriptions` (logD, logT1).
    geometry : tuple of float
        r [AU], delta [AU], alpha [rad].
    threads : int, optional
        Processes for the integration.

    Returns
    -------
    float
        Flux density in Jy.
    """
    one = pd.DataFrame({"logD": [fit["logD"]], "logT1": [fit["logT1"]]})
    return float(thermalFluxJy(model, lambd, one, geometry, threads)[0])


def _epsPerFilter(obs, eps: float, epsFixed: dict | None) -> np.ndarray:
    """
    Emissivity for each filter: `eps` for every filter not listed in `epsFixed`.

    Returns
    -------
    `~numpy.ndarray` (M)
    """
    values = np.full(len(obs.filterNames), float(eps))
    for name, value in (epsFixed or {}).items():
        values[obs.filterNames.index(name)] = value
    return values


def _modelFlux(
    model,
    obs,
    theta: np.ndarray,
    r,
    delta,
    alpha,
    G,
    epsFixed: dict | None = None,
    neowise: dict | None = None,
) -> np.ndarray:
    """
    IR model flux (thermal plus reflected) for parameters theta = (logD, logT1, x).

    By default x is eps: filters in `epsFixed` use their fixed emissivity instead of eps, and each
    filter's albedo follows Kirchhoff, p = (1 - eps) / Q. With `neowise` x is the albedo p of the
    listed bands, every filter has the fixed emissivity neowise["eps"], and the other filters have p = 0
    (the NEOWISE prescription, run5a of paper1).

    Returns
    -------
    `~numpy.ndarray` (N, M)
        Flux in W m^-3 for N observations by M filters.
    """
    logD, logT1, x = theta
    r, delta, alpha = (np.atleast_1d(v) for v in (r, delta, alpha))
    if neowise is None:
        epsF = _epsPerFilter(obs, x, epsFixed)
        pF = (1 - epsF) / calcQ(G)
    else:
        epsF = np.full(len(obs.filterNames), float(neowise["eps"]))
        pF = np.array([x if n in neowise["bands"] else 0.0 for n in obs.filterNames])
    # WISE-style bandpassLambda returns (M, N), SPHEREx-style (N, M); eps broadcasts along filters
    wiseLike = obs.acronym != "SPHEREx"
    e, p = (v[:, None] if wiseLike else v for v in (epsF, pF))
    flux = interpFluxLambdaObsWithSunlight(
        model, obs, r, delta, 10**logT1 / np.sqrt(r), 10**logD, alpha, e, p, G
    )
    return flux.T if wiseLike else flux


def fitEmcee(
    model,
    obs,
    data: pd.DataFrame,
    columnMapping: dict = Config.columnMapping,
    parameterPriors: dict = Config.parameterPriors,
    epsFixed: dict | None = None,
    neowise: dict | None = None,
    nwalkers: int = 32,
    nsteps: int = 3000,
    burnIn: int = 1000,
    seed: int = 0,
    verbose: bool = Config.verbose,
) -> pd.DataFrame:
    """
    Fit IR photometry of one object with a thermal model using emcee. This replaces the
    pymc3/theano `~atm.fit` for the single-epsilon, single-albedo (Kirchhoff, p = (1 - eps) / Q)
    case. Fit parameters are logD, logT1 (T_ss = 10**logT1 / sqrt(r)) and the IR eps, with
    uniform priors. Note that the IR eps is not the mm emissivity: eta is absorbed in logT1.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object; lambda tables are loaded for obs.filterQuadratureLambdas.
    obs : `~atm.obs.Observatory`
        Observatory object (e.g. WISE, SPHEREx).
    data : `~pandas.DataFrame`
        One row per epoch with columns for r_au, delta_au, alpha_rad, G, flux_si and fluxErr_si
        (SI, W m^-3), named as in columnMapping. NaN fluxes are ignored.
    columnMapping : dict, optional
        [Default = `~atm.Config.columnMapping`]
    parameterPriors : dict, optional
        Uniform bounds ("lower", "upper") for logD, logT1 and eps.
        [Default = `~atm.Config.parameterPriors`]
    epsFixed : dict, optional
        Filter name -> fixed emissivity (e.g. {"W3": 0.8, "W4": 0.98}, as in the paper1 runs). All other
        filters share the fitted eps, and every filter's albedo follows Kirchhoff.
    neowise : dict, optional
        NEOWISE prescription, {"eps": 0.9, "bands": ["W1", "W2"]}: all filters share the fixed emissivity, the
        albedo p of the listed bands is fitted instead of eps, and the other filters have p = 0. The third
        parameter (column) is then p. Overrides epsFixed.
    nwalkers : int, optional
        Number of emcee walkers.
    nsteps : int, optional
        Steps per walker, including burn-in.
    burnIn : int, optional
        Steps discarded.
    seed : int, optional
        Random seed.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    `~pandas.DataFrame`
        Flattened post-burn-in samples with columns logD, logT1, eps (p in NEOWISE mode), logprob; the
        acceptance fraction and the best-fit reduced chi^2 are stored in `.attrs`.
    """
    model.loadLambdaTables(obs.filterQuadratureLambdas, verbose=verbose)
    r, delta, alpha, G = (
        data[columnMapping[k]].values.astype(float)
        for k in ["r_au", "delta_au", "alpha_rad", "G"]
    )
    G = G[0]
    Y = data[columnMapping["flux_si"]].values.astype(float)
    Y_err = data[columnMapping["fluxErr_si"]].values.astype(float)
    good = np.isfinite(Y) & np.isfinite(Y_err)
    names = FIT_PARAMETERS[:2] + (["eps"] if neowise is None else ["p"])
    lo = np.array([parameterPriors[n]["lower"] for n in names])
    hi = np.array([parameterPriors[n]["upper"] for n in names])

    def chi2(theta: np.ndarray) -> float:
        T_ss = 10 ** theta[1] / np.sqrt(r)
        if (
            np.any(T_ss < T_SS_LIMITS[0])
            or np.any(T_ss > T_SS_LIMITS[1])
            or np.any(alpha < 0)
            or np.any(alpha > np.pi)
        ):
            return np.inf
        return float(
            (
                (
                    (
                        Y
                        - _modelFlux(
                            model, obs, theta, r, delta, alpha, G, epsFixed, neowise
                        )
                    )
                    / Y_err
                )[good]
                ** 2
            ).sum()
        )

    def logProb(theta: np.ndarray) -> float:
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return -0.5 * chi2(theta)

    rng = np.random.default_rng(seed)
    # Random search over the prior box restricted to valid T_ss, then polish
    logT1_lo, logT1_hi = (
        np.log10(T_SS_LIMITS[0] * np.sqrt(r.max())),
        np.log10(T_SS_LIMITS[1] * np.sqrt(r.min())),
    )
    box = np.column_stack(
        [
            rng.uniform(1, 6, 400),
            rng.uniform(logT1_lo, logT1_hi, 400),
            rng.uniform(0.5, 1, 400) if neowise is None else rng.uniform(0, 1, 400),
        ]
    )
    start = box[np.argmin([chi2(t) for t in box])]
    start = minimize(
        lambda t: -logProb(t) if np.isfinite(logProb(t)) else 1e30,
        start,
        method="Nelder-Mead",
    ).x
    p0 = start + 1e-3 * rng.standard_normal((nwalkers, 3))
    p0[:, 2] = np.minimum(p0[:, 2], hi[2] - 1e-6)
    sampler = emcee.EnsembleSampler(nwalkers, 3, logProb)
    sampler.run_mcmc(p0, nsteps, progress=verbose)
    flat = sampler.get_chain(discard=burnIn, flat=True)
    out = pd.DataFrame(flat, columns=names)
    out["logprob"] = sampler.get_log_prob(discard=burnIn, flat=True)
    out.attrs.update(
        acceptance=float(sampler.acceptance_fraction.mean()),
        chi2_best=chi2(flat[out["logprob"].idxmax()]),
        ndata=int(good.sum()),
    )
    return out


def predictMm(
    model,
    samples: pd.DataFrame,
    emissivity: EmissivityTable | str | Path,
    f_ghz: np.ndarray,
    r: float,
    delta: float,
    alpha: float,
    nDraws: int = 200,
    seed: int = 0,
) -> dict:
    """
    Predict the thermal mm spectrum from IR-fit posterior samples and a frequency-dependent emissivity.

    Each draw takes (logD, logT1) from the posterior; the IR eps is not used. If `emissivity` is a path to
    a Meteorites-TeraFlash NPZ file, each draw also gets a different posterior emissivity curve, so the
    band includes the emissivity uncertainty. Reflected sunlight is neglected at mm wavelengths.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model; tables at ghzGridToLambda-compatible wavelengths must have been built
        (scripts/build_mm_tables.py).
    samples : `~pandas.DataFrame`
        Output of `fitEmcee`.
    emissivity : `~atm.functions.EmissivityTable`, str or `~pathlib.Path`
        Emissivity table or the NPZ file to load.
    f_ghz : `~numpy.ndarray` (M)
        Frequencies in GHz, which must correspond to built tables.
    r, delta : float
        Heliocentric and observer distances in AU.
    alpha : float
        Phase angle in radians.
    nDraws : int, optional
        Number of posterior draws.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        f_ghz, F_nu (nDraws, M) in Jy, and its 16/50/84th percentiles (lo, med, hi).
    """
    lam = ghzToLambda(np.asarray(f_ghz, dtype=float))
    lam = np.array([float(f"{x / 1e-6:015.10f}e-06") for x in lam])
    model.loadLambdaTables(lam, verbose=False)
    rng = np.random.default_rng(seed)
    draws = samples.iloc[rng.choice(len(samples), nDraws, replace=False)]
    nEps = (
        len(np.load(emissivity)["e_draws"])
        if not isinstance(emissivity, EmissivityTable)
        else 0
    )
    F = np.empty((nDraws, len(lam)))
    for i, (logD, logT1) in enumerate(zip(draws["logD"], draws["logT1"])):
        table = (
            emissivity
            if nEps == 0
            else EmissivityTable.fromNpz(emissivity, draw=int(rng.integers(nEps)))
        )
        for j, lj in enumerate(lam):
            checkTableRange(model, lj, 10**logT1 / np.sqrt(r), alpha)
            F[i, j] = fluxLambdaToJy(
                interpFluxLambdaAtObs(
                    model, r, delta, lj, 10**logT1 / np.sqrt(r), 10**logD, alpha, table
                ),
                lj,
            )
    lo, med, hi = np.percentile(F, [16, 50, 84], axis=0)
    return {"f_ghz": np.asarray(f_ghz), "F_nu": F, "lo": lo, "med": med, "hi": hi}
