"""Forecast tools for Simons Observatory (SO) observations of an asteroid: observability, predicted flux and noise, and
the ability to distinguish emissivity hypotheses (an L5-like scattering e(nu) against a flat mm emissivity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constants import Constants
from .functions.emissivity import EmissivityTable, ghzListToLambda
from .functions.interpolate_flux_lambda import interpFluxLambdaAtObs
from .mmfit import checkTableRange, fluxLambdaToJy

__all__ = [
    "SO_DEPTH_MJY",
    "SO_GHZ",
    "SO_LAT_DEG",
    "SO_LON_DEG",
    "ALMA_GHZ",
    "distinguish",
    "distinguishN",
    "eCovariance",
    "getEphemerides",
    "observability",
    "posteriorCovariance",
    "survey_sigma",
    "thermalMjy",
    "transitAltitude",
]

# SO site: 22 deg 57' 31" S, 67 deg 47' 15" W (east longitude negative)
SO_LAT_DEG = -(22 + 57 / 60 + 31 / 3600)
SO_LON_DEG = -(67 + 47 / 60 + 15 / 3600)
SO_ELEVATION_KM = 5.2

SO_GHZ = np.array([30.0, 40.0, 90.0, 150.0, 220.0, 280.0])
# 1-sigma point-source depth per day of observing [mJy] in each band
SO_DEPTH_MJY = np.array([37.0, 25.0, 6.9, 8.3, 17.0, 49.0])

# ALMA receiver bands: nominal centre frequencies [GHz] (Band 10, 787-950 GHz, lies outside the L5 e(nu) table;
# Band 2 is in early science / commissioning)
ALMA_GHZ = {
    "B1": 43.0,
    "B2": 78.0,
    "B3": 100.0,
    "B4": 144.0,
    "B5": 187.0,
    "B6": 233.0,
    "B7": 343.0,
    "B8": 442.0,
    "B9": 661.0,
}

AU = Constants.ASTRONOMICAL_UNIT


def transitAltitude(
    dec_deg: float | np.ndarray, lat_deg: float = SO_LAT_DEG
) -> float | np.ndarray:
    """
    Altitude of an object at upper culmination (transit).

    Parameters
    ----------
    dec_deg : float or `~numpy.ndarray`
        Declination in degrees.
    lat_deg : float, optional
        Observatory latitude in degrees.
        [Default = `SO_LAT_DEG`]

    Returns
    -------
    float or `~numpy.ndarray`
        90 - |lat - dec| in degrees (the highest altitude the object reaches that day).
    """
    return 90.0 - np.abs(lat_deg - np.asarray(dec_deg, dtype=float))


def getEphemerides(
    horizonsId: str,
    start: str,
    stop: str,
    cache: Path | None = None,
    step: str = "1d",
) -> pd.DataFrame:
    """
    Daily topocentric ephemerides of a small body for the SO site from JPL Horizons (cached to CSV).

    Parameters
    ----------
    horizonsId : str
        Horizons small-body id (e.g. "8" for Flora).
    start, stop : str
        Start and stop dates "YYYY-MM-DD".
    cache : `~pathlib.Path`, optional
        CSV cache; read if it exists, written otherwise.
    step : str, optional
        Horizons step size.
        [Default = "1d"]

    Returns
    -------
    `~pandas.DataFrame`
        Columns jd, date, ra_deg, dec_deg, r_au, delta_au, alpha_deg, elong_deg (solar elongation).
    """
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache)
    from astroquery.jplhorizons import Horizons

    loc = {"lon": SO_LON_DEG, "lat": SO_LAT_DEG, "elevation": SO_ELEVATION_KM}
    e = Horizons(
        id=horizonsId,
        id_type="smallbody",
        location=loc,
        epochs={"start": start, "stop": stop, "step": step},
    ).ephemerides(quantities="1,19,20,23,24")
    df = pd.DataFrame(
        {
            "jd": np.array(e["datetime_jd"]),
            "date": [str(x) for x in e["datetime_str"]],
            "ra_deg": np.array(e["RA"]),
            "dec_deg": np.array(e["DEC"]),
            "r_au": np.array(e["r"]),
            "delta_au": np.array(e["delta"]),
            "alpha_deg": np.array(e["alpha"]),
            "elong_deg": np.array(e["elong"]),
        }
    )
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def observability(
    eph: pd.DataFrame,
    minAltitude: float = 30.0,
    dutyCycle: float = 0.5,
    lat_deg: float = SO_LAT_DEG,
    minElongation: float | None = None,
) -> pd.DataFrame:
    """
    Which days SO can observe the object, and the expected weight of each day.

    A day counts as observable if the object transits above `minAltitude` and then gets the full per-day depth;
    the survey duty cycle multiplies the expected exposure (each observable day has weight `dutyCycle`).

    Parameters
    ----------
    eph : `~pandas.DataFrame`
        Output of `getEphemerides`.
    minAltitude : float, optional
        Minimum transit altitude in degrees.
        [Default = 30]
    dutyCycle : float, optional
        Fraction of days on which the survey actually observes.
        [Default = 0.5]
    lat_deg : float, optional
        Observatory latitude.
    minElongation : float, optional
        Minimum solar elongation in degrees (Sun avoidance); needs the elong_deg column. Not applied by default.
        [Default = None]

    Returns
    -------
    `~pandas.DataFrame`
        eph plus transit_alt_deg, observable (bool) and weight (dutyCycle if observable, else 0).
    """
    out = eph.copy()
    out["transit_alt_deg"] = transitAltitude(out["dec_deg"].values, lat_deg)
    out["observable"] = out["transit_alt_deg"] >= minAltitude
    if minElongation is not None:
        out["observable"] &= out["elong_deg"] >= minElongation
    out["weight"] = np.where(out["observable"], dutyCycle, 0.0)
    return out


def thermalMjy(
    model,
    ghz: float,
    logD: float,
    logT1: float,
    r_au: np.ndarray,
    delta_au: np.ndarray,
    alpha_rad: np.ndarray,
) -> np.ndarray:
    """
    Thermal NEATM flux density (eps = 1, no reflected light) from lookup tables.

    Parameters
    ----------
    model : `~atm.models.NEATM`
        Model with the table at `ghz` loaded (key `ghzListToLambda`).
    ghz : float
        Frequency in GHz.
    logD, logT1 : float
        log10 diameter [m] and temperature parameter (T_ss = 10**logT1 / sqrt(r)).
    r_au, delta_au, alpha_rad : `~numpy.ndarray` (N)
        Geometry of N observations.

    Returns
    -------
    `~numpy.ndarray` (N)
        Flux density in mJy.
    """
    lam = float(ghzListToLambda(np.array([ghz]))[0])
    T = 10**logT1 / np.sqrt(r_au)
    checkTableRange(model, lam, T, alpha_rad)
    F = interpFluxLambdaAtObs(model, r_au, delta_au, lam, T, 10**logD, alpha_rad, 1.0)
    return 1e3 * fluxLambdaToJy(F, lam)


def posteriorCovariance(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian summary of the (logD, logT1) posterior of a WISE fit.

    Parameters
    ----------
    samples : `~pandas.DataFrame`
        Output of `~atm.mmfit.fitEmcee`.

    Returns
    -------
    mean : `~numpy.ndarray` (2)
    cov : `~numpy.ndarray` (2, 2)
    """
    x = samples[["logD", "logT1"]].values
    return x.mean(0), np.cov(x.T)


def survey_sigma(
    weights: np.ndarray, depth_mjy: np.ndarray = SO_DEPTH_MJY
) -> np.ndarray:
    """
    1-sigma noise of each daily measurement in each band.

    Parameters
    ----------
    weights : `~numpy.ndarray` (T)
        Expected exposure weight of each day (0 if the object cannot be observed, the duty cycle otherwise); the noise
        scales as depth / sqrt(weight).
    depth_mjy : `~numpy.ndarray` (B), optional
        1-sigma depth per full day in each band.
        [Default = `SO_DEPTH_MJY`]

    Returns
    -------
    `~numpy.ndarray` (T, B)
        Noise in mJy, infinite where weight is zero.
    """
    w = np.asarray(weights, dtype=float)[:, None]
    with np.errstate(divide="ignore"):
        return np.where(
            w > 0,
            np.asarray(depth_mjy)[None, :] / np.sqrt(np.where(w > 0, w, 1.0)),
            np.inf,
        )


def _invVar(sigma: np.ndarray) -> np.ndarray:
    """1 / sigma^2, zero where sigma is infinite."""
    return np.where(
        np.isfinite(sigma), 1.0 / np.where(np.isfinite(sigma), sigma, 1.0) ** 2, 0.0
    )


def _fit(
    data: np.ndarray,
    th: np.ndarray,
    kernel: np.ndarray,
    w: np.ndarray,
    priorInv: np.ndarray,
    e0: np.ndarray,
    freqIdx: np.ndarray,
    covL5: np.ndarray | None,
    groups: list,
    hypothesis: str,
):
    """
    Minimise chi^2 of one hypothesis over its free and nuisance parameters.

    Parameters are f (flat hypothesis only), dtheta (2), one log-scale nuisance per entry of `groups`, and the
    amplitudes of the significant eigenmodes of the L5 e covariance (L5 hypothesis with a covariance only). All entries are 1-d
    arrays over the N measurements, except kernel (N, 2).
    """
    N = len(th)
    K = len(groups)
    masks = np.array([g[0] for g in groups], dtype=float).reshape(K, N)
    gInv = np.array([1.0 / g[1] ** 2 for g in groups])
    # The L5 e(nu) posterior is a smooth curve, so its covariance is nearly singular; parametrise the nuisance by the
    # significant eigenmodes (de = Ls @ a with a ~ N(0, 1)) instead of inverting the covariance.
    Ls = None
    if covL5 is not None and hypothesis == "L5":
        lam, V = np.linalg.eigh(np.asarray(covL5, dtype=float))
        keep = lam > 1e-10 * lam.max()
        Ls = V[:, keep] * np.sqrt(lam[keep])
    F = Ls.shape[1] if Ls is not None else 0
    off = 1 if hypothesis == "flat" else 0

    def fn(p: np.ndarray) -> float:
        dth, ds = p[off : off + 2], p[off + 2 : off + 2 + K]
        if hypothesis == "flat":
            e = p[0]
        else:
            a = p[off + 2 + K :]
            e = e0 + ((Ls @ a)[freqIdx] if F else 0.0)
        m = e * th * np.exp(kernel @ dth + ds @ masks)
        c = float(
            (w * (data - m) ** 2).sum() + dth @ priorInv @ dth + (gInv * ds**2).sum()
        )
        return c + (float(a @ a) if F else 0.0)

    n = off + 2 + K + F
    starts = (
        [np.r_[f0, np.zeros(n - 1)] for f0 in (0.6, 0.85, 1.1)]
        if hypothesis == "flat"
        else [np.zeros(n)]
    )
    return min((minimize(fn, s0, method="BFGS") for s0 in starts), key=lambda o: o.fun)


def distinguishN(
    th: np.ndarray,
    kernel: np.ndarray,
    sigma: np.ndarray,
    priorCov: np.ndarray,
    e0: np.ndarray,
    freqIdx: np.ndarray | None = None,
    truth: str = "L5",
    fTrue: float | None = None,
    covL5: np.ndarray | None = None,
    groups: list | None = None,
) -> dict:
    """
    Expected (Asimov) test between an L5-like emissivity and a flat mm emissivity for an arbitrary set of N
    measurements, marginalising over the IR thermal model and over per-instrument calibration scales.

    Every measurement i predicts flux_i = e_i * Th_i * exp(kernel_i . dtheta + sum_k mask_ki ds_k): the thermal flux
    Th (eps_IR included) at the central IR-fit parameters, the IR-model nuisance dtheta (Gaussian prior `priorCov`)
    and optional multiplicative calibration nuisances ds_k (one per entry of `groups`, each applying to the
    measurements selected by its mask, with a Gaussian prior of the given fractional width). The L5 hypothesis has
    the fixed e_i = e0_i (optionally with the Gaussian uncertainty `covL5`, one entry per distinct frequency, see
    `freqIdx`); the flat hypothesis has e_i = f with f free.

    The statistic is dchi2 = chi2_min(wrong hypothesis) given data equal to the true model (evaluated at its own
    central values), and the significance is sqrt(dchi2).

    Parameters
    ----------
    th : `~numpy.ndarray` (N)
        eps_IR x thermal flux at eps = 1 at the central IR-fit parameters [mJy].
    kernel : `~numpy.ndarray` (N, 2)
        d ln Th / d(logD, logT1).
    sigma : `~numpy.ndarray` (N)
        1-sigma noise of each measurement [mJy], infinite to drop one.
    priorCov : `~numpy.ndarray` (2, 2)
        Covariance of (logD, logT1) from the IR fit.
    e0 : `~numpy.ndarray` (N)
        Central L5 e at each measurement's frequency.
    freqIdx : `~numpy.ndarray` (N), optional
        Index of each measurement's frequency into the rows of `covL5`.
    truth : {"L5", "flat"}, optional
        Which hypothesis generates the data (data = fTrue * th for "flat").
        [Default = "L5"]
    fTrue : float, optional
        The flat emissivity of the truth when truth == "flat".
    covL5 : `~numpy.ndarray` (F, F), optional
        Covariance of the L5 e at the F distinct frequencies (nuisance of the L5 hypothesis).
    groups : list of (mask, fractional sigma), optional
        Calibration-scale nuisances: boolean mask over the N measurements and the 1-sigma fractional prior.

    Returns
    -------
    dict
        dchi2, significance, the best-fit IR-model shifts dtheta and, for truth == "L5", the best-fit flat f.
    """
    th, sigma, e0 = (np.asarray(x, dtype=float) for x in (th, sigma, e0))
    w = _invVar(sigma)
    priorInv = np.linalg.inv(priorCov)
    freqIdx = (
        np.zeros(len(th), dtype=int)
        if freqIdx is None
        else np.asarray(freqIdx, dtype=int)
    )
    groups = groups or []
    if truth == "L5":
        res = _fit(e0 * th, th, kernel, w, priorInv, e0, freqIdx, covL5, groups, "flat")
        return {
            "dchi2": float(res.fun),
            "significance": float(np.sqrt(max(res.fun, 0.0))),
            "f_best": float(res.x[0]),
            "dtheta": res.x[1:3],
        }
    if truth == "flat":
        if fTrue is None:
            raise ValueError("fTrue is required when truth == 'flat'")
        res = _fit(
            fTrue * th, th, kernel, w, priorInv, e0, freqIdx, covL5, groups, "L5"
        )
        return {
            "dchi2": float(res.fun),
            "significance": float(np.sqrt(max(res.fun, 0.0))),
            "dtheta": res.x[:2],
        }
    raise ValueError("truth must be 'L5' or 'flat'")


def distinguish(
    th: np.ndarray,
    kernel: np.ndarray,
    sigma: np.ndarray,
    priorCov: np.ndarray,
    e0: np.ndarray,
    truth: str = "L5",
    fTrue: float | None = None,
    covL5: np.ndarray | None = None,
) -> dict:
    """
    `distinguishN` for a regular (T, B) grid of measurements: T days (or epochs) in the same B bands, with no
    calibration nuisances.

    Parameters
    ----------
    th : `~numpy.ndarray` (T, B)
        eps_IR x thermal flux (eps = 1) at the central IR-fit parameters [mJy].
    kernel : `~numpy.ndarray` (T, B, 2)
        d ln Th / d(logD, logT1).
    sigma : `~numpy.ndarray` (T, B)
        1-sigma noise of each measurement [mJy], infinite for days that are not observed.
    priorCov : `~numpy.ndarray` (2, 2)
        Covariance of (logD, logT1) from the IR fit.
    e0 : `~numpy.ndarray` (B)
        Central e_b of the L5 hypothesis.
    truth, fTrue, covL5 :
        As in `distinguishN`; covL5 is (B, B).

    Returns
    -------
    dict
        As in `distinguishN`.
    """
    T, B = th.shape
    return distinguishN(
        th.ravel(),
        kernel.reshape(T * B, 2),
        sigma.ravel(),
        priorCov,
        np.tile(np.asarray(e0, dtype=float), T),
        freqIdx=np.tile(np.arange(B), T),
        truth=truth,
        fTrue=fTrue,
        covL5=covL5,
    )


def eCovariance(npz: str | Path, ghz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Central value and covariance of an emissivity table's posterior draws at the requested frequencies.

    Parameters
    ----------
    npz : str or `~pathlib.Path`
        File written by Meteorites-TeraFlash `export_emissivity.py` (arrays f_ghz, e, e_draws).
    ghz : `~numpy.ndarray` (B)
        Frequencies in GHz, inside the tabulated range.

    Returns
    -------
    e0 : `~numpy.ndarray` (B)
        Best-fit e at ghz.
    cov : `~numpy.ndarray` (B, B)
        Covariance of the posterior draws at ghz.
    """
    d = np.load(npz)
    e0 = EmissivityTable(d["f_ghz"], d["e"])(299792458.0 / (ghz * 1e9))
    draws = np.array(
        [
            EmissivityTable(d["f_ghz"], row)(299792458.0 / (ghz * 1e9))
            for row in d["e_draws"]
        ]
    )
    return e0, np.cov(draws.T)
