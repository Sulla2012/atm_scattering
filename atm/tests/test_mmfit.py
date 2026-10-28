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
