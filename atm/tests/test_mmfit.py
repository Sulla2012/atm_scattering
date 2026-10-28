import numpy as np
import pandas as pd
import pytest

from ..config import Config
from ..mmfit import _modelFlux, fitEmcee
from ..models import NEATM
from ..obs import WISE

TRUTH = np.array([3.0, np.log10(300 * np.sqrt(2)), 0.9])


@pytest.fixture(scope="module")
def synthetic():
    obs, model = WISE(), NEATM()
    model.loadLambdaTables(obs.filterQuadratureLambdas, verbose=False)
    r, delta, alpha = (
        np.array([2.0, 2.0, 2.1]),
        np.array([1.2, 1.1, 1.0]),
        np.array([0.4, 0.5, 0.6]),
    )
    Y = _modelFlux(model, obs, TRUTH, r, delta, alpha, 0.15)
    err = 0.05 * Y
    Y = Y + err * np.random.default_rng(1).standard_normal(Y.shape)
    df = pd.DataFrame({"r_au": r, "delta_au": delta, "alpha_rad": alpha, "G": 0.15})
    for i, name in enumerate(obs.filterNames):
        df[f"flux_{name}_si"], df[f"fluxErr_{name}_si"] = Y[:, i], err[:, i]
    cm = dict(Config.columnMapping)
    return model, obs, df, cm


def test_fitRecoversTruth(synthetic):
    model, obs, df, cm = synthetic
    s = fitEmcee(
        model,
        obs,
        df,
        columnMapping=cm,
        nwalkers=16,
        nsteps=500,
        burnIn=200,
        verbose=False,
    )
    for p, t, tol in zip(["logD", "logT1", "eps"], TRUTH, [0.05, 0.02, 0.05]):
        assert abs(s[p].median() - t) < tol
    assert s.attrs["chi2_best"] < 3 * s.attrs["ndata"]


def test_nanFluxIgnored(synthetic):
    model, obs, df, cm = synthetic
    df = df.copy()
    df.loc[0, "flux_W1_si"] = np.nan
    s = fitEmcee(
        model,
        obs,
        df,
        columnMapping=cm,
        nwalkers=16,
        nsteps=200,
        burnIn=100,
        verbose=False,
    )
    assert s.attrs["ndata"] == 11


def test_modelFluxMatchesSingleEpoch(synthetic):
    """Each epoch/band of _modelFlux must equal the model evaluated for that epoch alone."""
    model, obs, _, _ = synthetic
    r, delta, alpha = (
        np.array([2.0, 2.2, 2.4]),
        np.array([1.2, 1.4, 1.6]),
        np.array([0.1, 0.3, 0.5]),
    )
    all_ = _modelFlux(model, obs, TRUTH, r, delta, alpha, 0.15)
    assert all_.shape == (3, 4)
    for i in range(3):
        one = _modelFlux(
            model, obs, TRUTH, r[i : i + 1], delta[i : i + 1], alpha[i : i + 1], 0.15
        )
        np.testing.assert_allclose(all_[i], one[0])


def test_epsFixedChangesOnlyThoseBands(synthetic):
    model, obs, _, _ = synthetic
    r, delta, alpha = np.array([2.0]), np.array([1.2]), np.array([0.3])
    base = _modelFlux(model, obs, TRUTH, r, delta, alpha, 0.15)
    fixed = _modelFlux(model, obs, TRUTH, r, delta, alpha, 0.15, {"W4": 0.5})
    np.testing.assert_allclose(base[0, :3], fixed[0, :3])
    assert not np.isclose(base[0, 3], fixed[0, 3], rtol=1e-3, atol=0)


def test_neowiseModeUsesFixedEpsAndBandAlbedo(synthetic):
    """eps = 0.9 in all bands, albedo only in W1/W2: W3/W4 are pure thermal, W1/W2 gain reflected light with p."""
    model, obs, _, _ = synthetic
    r, delta, alpha = np.array([2.0]), np.array([1.2]), np.array([0.3])
    cfg = {"eps": 0.9, "bands": ["W1", "W2"]}
    th = np.array([3.0, TRUTH[1], 0.0])
    noRefl = _modelFlux(model, obs, th, r, delta, alpha, 0.15, neowise=cfg)
    withRefl = _modelFlux(
        model, obs, th + [0, 0, 0.3], r, delta, alpha, 0.15, neowise=cfg
    )
    np.testing.assert_allclose(noRefl[0, 2:], withRefl[0, 2:])
    assert np.all(withRefl[0, :2] > noRefl[0, :2])
    # p = 0 thermal-only flux equals the Kirchhoff model's thermal part at the same eps (eps = 0.9, p = 0)
    kirch = _modelFlux(
        model, obs, np.array([3.0, TRUTH[1], 0.9]), r, delta, alpha, 0.15
    )
    assert np.all(kirch[0] >= noRefl[0] * (1 - 1e-12))


def test_neowiseFitRecoversTruth(synthetic):
    model, obs, df, cm = synthetic
    cfg = {"eps": 0.9, "bands": ["W1", "W2"]}
    r, delta, alpha = df["r_au"].values, df["delta_au"].values, df["alpha_rad"].values
    truth = np.array([3.0, TRUTH[1], 0.25])
    Y = _modelFlux(model, obs, truth, r, delta, alpha, 0.15, neowise=cfg)
    d = df.copy()
    for i, name in enumerate(obs.filterNames):
        d[f"flux_{name}_si"], d[f"fluxErr_{name}_si"] = Y[:, i], 0.03 * Y[:, i]
    s = fitEmcee(
        model,
        obs,
        d,
        columnMapping=cm,
        neowise=cfg,
        nwalkers=16,
        nsteps=500,
        burnIn=200,
        verbose=False,
    )
    assert list(s.columns[:3]) == ["logD", "logT1", "p"]
    for p, t, tol in zip(["logD", "logT1", "p"], truth, [0.05, 0.02, 0.1]):
        assert abs(s[p].median() - t) < tol
