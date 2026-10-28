from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import Constants

__all__ = [
    "EmissivityTable",
    "evalEmissivity",
    "ghzGridToLambda",
    "ghzToLambda",
    "lambdaToGHz",
]

C = Constants.LIGHTSPEED


def lambdaToGHz(lambd: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength to frequency.

    Parameters
    ----------
    lambd : float or `~numpy.ndarray`
        Wavelength in m.

    Returns
    -------
    float or `~numpy.ndarray`
        Frequency in GHz.
    """
    return C / np.asarray(lambd, dtype=float) / 1e9


def ghzToLambda(f_ghz: float | np.ndarray) -> float | np.ndarray:
    """
    Convert frequency to wavelength.

    Parameters
    ----------
    f_ghz : float or `~numpy.ndarray`
        Frequency in GHz.

    Returns
    -------
    float or `~numpy.ndarray`
        Wavelength in m.
    """
    return C / (np.asarray(f_ghz, dtype=float) * 1e9)


def ghzGridToLambda(
    fmin: float = 50.0, fmax: float = 1500.0, df: float = 25.0
) -> np.ndarray:
    """
    Wavelengths of a regular frequency grid, rounded to the precision of the table
    file names (so they are valid keys for `~atm.models.Model.loadLambdaTables`). Use this to build NEATM lookup tables
    so that the wavelength keys match those used later when evaluating fluxes.

    Parameters
    ----------
    fmin, fmax : float, optional
        Frequency range in GHz (inclusive).
        [Default = 50, 1500]
    df : float, optional
        Frequency step in GHz.
        [Default = 25]

    Returns
    -------
    `~numpy.ndarray`
        Wavelengths in m, in order of increasing frequency.
    """
    lambd = ghzToLambda(np.arange(fmin, fmax + df / 2, df))
    return np.array([float(f"{x / 1e-6:015.10f}e-06") for x in lambd])


class EmissivityTable:
    """
    Frequency-dependent effective emissivity, callable on wavelength in m so that
    it can be passed anywhere in `atm` that accepts `eps`.

    Parameters
    ----------
    f_ghz : `~numpy.ndarray` (N)
        Frequencies in GHz (any order).
    eps : `~numpy.ndarray` (N)
        Emissivity at f_ghz.
    """

    def __init__(self, f_ghz: np.ndarray, eps: np.ndarray) -> None:
        order = np.argsort(f_ghz)
        self.f_ghz = np.asarray(f_ghz, dtype=float)[order]
        self.eps = np.asarray(eps, dtype=float)[order]

    @classmethod
    def fromNpz(cls, path: str | Path, draw: str | int = "best") -> EmissivityTable:
        """
        Load a file written by Meteorites-TeraFlash `export_emissivity.py`.

        Parameters
        ----------
        path : str or `~pathlib.Path`
            Path to the .npz file.
        draw : {"best", "median"} or int, optional
            Which emissivity curve to use: the best fit, the median of the
            posterior draws, or the i-th posterior draw (to propagate uncertainty).
            [Default = "best"]

        Returns
        -------
        `~atm.functions.EmissivityTable`
        """
        d = np.load(path)
        if draw == "best":
            eps = d["e"]
        elif draw == "median":
            eps = np.median(d["e_draws"], axis=0)
        else:
            eps = d["e_draws"][int(draw)]
        return cls(d["f_ghz"], eps)

    def __call__(self, lambd: float | np.ndarray) -> float | np.ndarray:
        """
        Interpolate the emissivity at wavelength lambd.

        Parameters
        ----------
        lambd : float or `~numpy.ndarray` (N)
            Wavelength in m.

        Returns
        -------
        float or `~numpy.ndarray` (N)
            Emissivity.

        Raises
        ------
        ValueError
            If any wavelength lies outside the tabulated frequency range.
        """
        f = lambdaToGHz(lambd)
        tol = 1e-6  # absorbs the rounding of table-key wavelengths
        if np.any(f < self.f_ghz[0] * (1 - tol)) or np.any(
            f > self.f_ghz[-1] * (1 + tol)
        ):
            raise ValueError(
                f"Frequency outside emissivity table range [{self.f_ghz[0]:.1f}, {self.f_ghz[-1]:.1f}] GHz."
            )
        return np.interp(f, self.f_ghz, self.eps)

    def bandAverage(
        self, lambd: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        """
        Weighted mean emissivity, e.g. the scalar needed by `~atm.functions.calcTss`.

        Parameters
        ----------
        lambd : `~numpy.ndarray` (N)
            Wavelengths in m.
        weights : `~numpy.ndarray` (N), optional
            Weights (e.g. filter response times a Planck spectrum).
            [Default = uniform]

        Returns
        -------
        float
            Weighted mean emissivity.
        """
        return float(np.average(self(lambd), weights=weights))


def evalEmissivity(eps, lambd):
    """
    Evaluate eps at lambd if it is callable (e.g. an `~atm.functions.EmissivityTable`),
    otherwise return it unchanged.

    Parameters
    ----------
    eps : float, `~numpy.ndarray` or callable
        Emissivity, or a function of wavelength in m.
    lambd : float or `~numpy.ndarray` (N)
        Wavelength in m.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Emissivity.
    """
    return eps(lambd) if callable(eps) else eps
