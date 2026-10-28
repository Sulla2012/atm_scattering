import numpy as np
import pytest

from ..forecast import (
    SO_DEPTH_MJY,
    SO_LAT_DEG,
    distinguish,
    observability,
    survey_sigma,
    transitAltitude,
)

import pandas as pd


def test_transitAltitude():
    assert transitAltitude(SO_LAT_DEG) == pytest.approx(90.0)  # transits at the zenith
    assert transitAltitude(SO_LAT_DEG + 30.0) == pytest.approx(60.0)
    assert transitAltitude(SO_LAT_DEG - 30.0) == pytest.approx(60.0)
    assert transitAltitude(90.0) == pytest.approx(90.0 - (90.0 - SO_LAT_DEG))


def test_observabilityThreshold():
    eph = pd.DataFrame({"dec_deg": [SO_LAT_DEG, SO_LAT_DEG + 59.9, SO_LAT_DEG + 60.1]})
    out = observability(eph, minAltitude=30.0, dutyCycle=0.5)
    assert list(out["observable"]) == [True, True, False]
    assert list(out["weight"]) == [0.5, 0.5, 0.0]


def test_surveySigma():
    s = survey_sigma(np.array([1.0, 0.5, 0.0]))
    np.testing.assert_allclose(s[0], SO_DEPTH_MJY)
    np.testing.assert_allclose(s[1], SO_DEPTH_MJY / np.sqrt(0.5))
    assert np.all(np.isinf(s[2]))


def _toy(T=40, B=6, scale=100.0):
    rng = np.random.default_rng(0)
    th = (
        scale
        * (1 + 0.3 * rng.random((T, 1)))
        * (np.array([30, 40, 90, 150, 220, 280.0]) / 150) ** 2
    )
    th = np.broadcast_to(th, (T, B)).copy() if th.shape[1] != B else th
    kernel = np.stack([np.full((T, B), 2 * np.log(10)), np.full((T, B), 0.8)], axis=-1)
    return th, kernel


def test_flatTruthAgainstFlatFitIsZero():
    """If the data are exactly flat and the truth is L5 with a flat e, the flat hypothesis fits perfectly."""
    th, kernel = _toy()
    sigma = np.full(th.shape, 1.0)
    out = distinguish(
        th, kernel, sigma, np.diag([1e-4, 1e-4]), e0=np.full(6, 0.8), truth="L5"
    )
    assert out["dchi2"] == pytest.approx(0.0, abs=1e-3)
    assert out["f_best"] == pytest.approx(0.8, abs=1e-3)


def test_tiltedL5MatchesClosedFormWithoutNuisance():
    """With the nuisance parameters pinned (tiny prior), dchi2 equals the weighted least squares of a flat f."""
    th, kernel = _toy()
    e0 = np.array([0.90, 0.86, 0.845, 0.88, 0.90, 0.92])
    sigma = np.full(th.shape, 0.5)
    out = distinguish(th, kernel, sigma, np.diag([1e-10, 1e-10]), e0=e0, truth="L5")
    d = e0[None, :] * th
    w = 1 / sigma**2
    f = (w * d * th).sum() / (w * th**2).sum()
    chi = (w * (d - f * th) ** 2).sum()
    assert out["dchi2"] == pytest.approx(chi, rel=1e-4)
    assert out["f_best"] == pytest.approx(f, rel=1e-4)


def test_looseNuisanceCannotMimicTilt_butRemovesNormalisation():
    """A loose prior on logD absorbs a pure normalisation difference between truth and L5, not a tilt."""
    th, kernel = _toy()
    sigma = np.full(th.shape, 0.5)
    e0 = np.full(6, 0.85)
    tight = distinguish(
        th, kernel, sigma, np.diag([1e-10, 1e-10]), e0, truth="flat", fTrue=0.6
    )
    loose = distinguish(
        th, kernel, sigma, np.diag([1.0, 1.0]), e0, truth="flat", fTrue=0.6
    )
    assert tight["dchi2"] > 100 * max(
        loose["dchi2"], 1e-6
    )  # the nuisance absorbs the normalisation


def test_requiresFTrue():
    th, kernel = _toy()
    with pytest.raises(ValueError):
        distinguish(th, kernel, np.ones(th.shape), np.eye(2), np.ones(6), truth="flat")


def test_observabilitySolarElongationCut():
    eph = pd.DataFrame({"dec_deg": [SO_LAT_DEG] * 3, "elong_deg": [20.0, 50.0, 120.0]})
    assert list(observability(eph)["observable"]) == [True, True, True]
    assert list(observability(eph, minElongation=45.0)["observable"]) == [
        False,
        True,
        True,
    ]


def test_calibrationScaleNuisanceWeakensTheTest():
    """Giving one instrument a free calibration scale makes L5 vs flat harder to separate, never easier."""
    from ..forecast import distinguishN

    ghz = np.array([40.0, 90.0, 150.0, 220.0, 345.0])
    e0 = np.array([0.86, 0.845, 0.88, 0.903, 0.93])
    th = 100.0 * (ghz / 150.0) ** 2
    kernel = np.stack([np.full(5, 2 * np.log(10)), np.full(5, 0.8)], axis=-1)
    sigma = 0.01 * th * e0
    cov = np.diag([1e-8, 1e-8])
    base = distinguishN(th, kernel, sigma, cov, e0, truth="L5")
    mask = np.array(
        [False, False, False, True, True]
    )  # the two highest bands share a loose scale
    loose = distinguishN(th, kernel, sigma, cov, e0, truth="L5", groups=[(mask, 0.1)])
    pinned = distinguishN(th, kernel, sigma, cov, e0, truth="L5", groups=[(mask, 1e-6)])
    assert 0 < loose["significance"] < base["significance"]
    assert pinned["significance"] == pytest.approx(base["significance"], rel=1e-3)


def test_distinguishNMatchesGridWrapper():
    from ..forecast import distinguishN

    th, kernel = _toy()
    e0 = np.array([0.90, 0.86, 0.845, 0.88, 0.90, 0.92])
    sigma = np.full(th.shape, 0.5)
    cov = np.diag([1e-4, 1e-4])
    a = distinguish(th, kernel, sigma, cov, e0, truth="L5")
    T, B = th.shape
    b = distinguishN(
        th.ravel(),
        kernel.reshape(T * B, 2),
        sigma.ravel(),
        cov,
        np.tile(e0, T),
        truth="L5",
    )
    assert a["significance"] == pytest.approx(b["significance"], rel=1e-6)


def test_l5CovarianceNuisanceIsMarginalisedOverUnusedFrequencies():
    """A near-singular covariance over extra, unused frequencies must give the same result as its marginal."""
    from ..forecast import distinguishN

    g = np.geomspace(20, 700, 15)
    modes = np.stack([np.ones_like(g), np.log(g), np.log(g) ** 2], axis=1)
    A = modes * np.array([0.01, 0.004, 0.001])
    C15 = A @ A.T  # rank 3 over 15 frequencies
    used = np.array([3, 5, 7, 9, 11, 13])
    th = np.full(6, 50.0)
    kernel = np.stack([np.full(6, 2 * np.log(10)), np.full(6, 0.8)], axis=-1)
    e0 = 0.85 + 0.02 * np.log(g[used] / g[used][0])
    sigma = 0.02 * th
    cov = np.diag([1e-3, 1e-3])
    full = distinguishN(
        th, kernel, sigma, cov, e0, freqIdx=used, truth="flat", fTrue=0.7, covL5=C15
    )
    marg = distinguishN(
        th,
        kernel,
        sigma,
        cov,
        e0,
        freqIdx=np.arange(6),
        truth="flat",
        fTrue=0.7,
        covL5=C15[np.ix_(used, used)],
    )
    assert full["significance"] == pytest.approx(marg["significance"], rel=2e-2)
