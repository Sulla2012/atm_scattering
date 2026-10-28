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
    "distinguish",
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


def _fitFlat(data, th, kernel, w, priorInv):
    """Minimise chi^2 of the flat hypothesis (free f, nuisance dtheta); returns the optimiser result."""

    def fn(p: np.ndarray) -> float:
        m = p[0] * th * np.exp(kernel @ p[1:])
        return float((w * (data - m) ** 2).sum() + p[1:] @ priorInv @ p[1:])

    starts = [np.r_[f0, 0.0, 0.0] for f0 in (0.6, 0.85, 1.1)]
    return min((minimize(fn, s0, method="BFGS") for s0 in starts), key=lambda o: o.fun)


def _fitL5(data, th, kernel, w, priorInv, e0, covL5):
    """Minimise chi^2 of the L5 hypothesis (fixed e_b up to the Gaussian uncertainty covL5, nuisance dtheta)."""
    B = len(e0)
    eInv = np.linalg.inv(covL5) if covL5 is not None else None

    def fn(p: np.ndarray) -> float:
        dth, de = p[:2], p[2:]
        m = (e0 + de) * th * np.exp(kernel @ dth)
        c = float((w * (data - m) ** 2).sum() + dth @ priorInv @ dth)
        return c + (float(de @ eInv @ de) if eInv is not None else 0.0)

    free = B if covL5 is not None else 0
    if free == 0:  # without an L5 uncertainty only dtheta is free
        res = minimize(lambda q: fn(np.r_[q, np.zeros(B)]), np.zeros(2), method="BFGS")
        return res
    return minimize(fn, np.zeros(2 + B), method="BFGS")


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
    Expected (Asimov, noiseless-data) test between an L5-like emissivity and a flat mm emissivity, marginalising
    over the uncertainty of the IR thermal model.

    Both hypotheses predict flux(t, b) = e_b * Th(t, b) * exp(kernel . dtheta), where Th is the thermal flux (eps_IR
    included) at the central IR-fit parameters and dtheta = (d logD, d logT1) are nuisance parameters with the
    Gaussian prior `priorCov`. The L5 hypothesis has the fixed e_b = e0 (optionally with the Gaussian uncertainty
    `covL5`); the flat hypothesis has e_b = f for every band with f free. The statistic is
    dchi2 = chi2_min(wrong hypothesis) given data equal to the true model, and the significance is sqrt(dchi2).

    Parameters
    ----------
    th : `~numpy.ndarray` (T, B)
        eps_IR x thermal flux (eps = 1) at the central IR-fit parameters [mJy].
    kernel : `~numpy.ndarray` (T, B, 2)
        d ln Th / d(logD, logT1).
    sigma : `~numpy.ndarray` (T, B)
        1-sigma noise of each daily measurement [mJy], infinite for days that are not observed.
    priorCov : `~numpy.ndarray` (2, 2)
        Covariance of (logD, logT1) from the IR fit.
    e0 : `~numpy.ndarray` (B)
        Central e_b of the L5 hypothesis.
    truth : {"L5", "flat"}, optional
        Which hypothesis generates the data. For "flat" the data are fTrue * th.
        [Default = "L5"]
    fTrue : float, optional
        The flat emissivity of the truth when truth == "flat".
    covL5 : `~numpy.ndarray` (B, B), optional
        Covariance of the L5 e_b (nuisance of the L5 hypothesis).

    Returns
    -------
    dict
        dchi2, significance, and for truth == "L5" the best-fit flat f; the best-fit nuisance shifts.
    """
    w = _invVar(sigma)
    priorInv = np.linalg.inv(priorCov)
    if truth == "L5":
        data = e0[None, :] * th
        res = _fitFlat(data, th, kernel, w, priorInv)
        return {
            "dchi2": float(res.fun),
            "significance": float(np.sqrt(max(res.fun, 0.0))),
            "f_best": float(res.x[0]),
            "dtheta": res.x[1:],
        }
    if truth == "flat":
        if fTrue is None:
            raise ValueError("fTrue is required when truth == 'flat'")
        data = fTrue * th
        res = _fitL5(data, th, kernel, w, priorInv, e0, covL5)
        return {
            "dchi2": float(res.fun),
            "significance": float(np.sqrt(max(res.fun, 0.0))),
            "dtheta": res.x[:2],
        }
    raise ValueError("truth must be 'L5' or 'flat'")


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
