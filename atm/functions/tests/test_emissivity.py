import numpy as np
import pytest
from numpy import testing as test

from ...constants import Constants
from ..emissivity import EmissivityTable, evalEmissivity, ghzToLambda, lambdaToGHz
from ..interpolate_flux_lambda import interpFluxLambdaAtObs

c = Constants.LIGHTSPEED


class _StubModel:
    """Model returning a flux independent of T_ss and alpha."""

    def interpTotalFluxLambdaEmittedToObs(self, lambd, T_ss, alpha):
        return np.ones_like(lambd, dtype=float)


@pytest.fixture
def npz(tmp_path):
    f = np.array([50.0, 100.0, 200.0, 400.0])
    draws = np.vstack([f * 0 + 0.5, f * 0 + 0.7])
    path = tmp_path / "eps.npz"
    np.savez(path, f_ghz=f, e=f / 1000.0, e_draws=draws)
    return path


def test_unitRoundTrip():
    f = np.array([50.0, 750.0, 1500.0])
    test.assert_allclose(lambdaToGHz(ghzToLambda(f)), f)
    test.assert_allclose(ghzToLambda(300.0), c / 300e9)


def test_interpolation(npz):
    table = EmissivityTable.fromNpz(npz)
    test.assert_allclose(table(ghzToLambda(150.0)), 0.15)
    # descending wavelength / unsorted input
    lam = ghzToLambda(np.array([400.0, 50.0, 100.0]))
    test.assert_allclose(table(lam), [0.4, 0.05, 0.1])


def test_drawSelection(npz):
    lam = ghzToLambda(100.0)
    test.assert_allclose(EmissivityTable.fromNpz(npz, draw="median")(lam), 0.6)
    test.assert_allclose(EmissivityTable.fromNpz(npz, draw=1)(lam), 0.7)


def test_outOfRangeRaises(npz):
    table = EmissivityTable.fromNpz(npz)
    with pytest.raises(ValueError):
        table(ghzToLambda(10.0))
    with pytest.raises(ValueError):
        table(ghzToLambda(1000.0))


def test_bandAverage(npz):
    table = EmissivityTable.fromNpz(npz)
    lam = ghzToLambda(np.array([100.0, 200.0]))
    test.assert_allclose(table.bandAverage(lam), 0.15)
    test.assert_allclose(table.bandAverage(lam, weights=[3, 1]), 0.125)


def test_constantTableMatchesScalar():
    f = np.array([50.0, 400.0])
    table = EmissivityTable(f, np.full(2, 0.9))
    lam = ghzToLambda(np.array([100.0, 200.0, 300.0]))
    args = (_StubModel(), 1.0, 1.0, lam, 300.0, 1e3, 0.1)
    test.assert_allclose(
        interpFluxLambdaAtObs(*args, table), interpFluxLambdaAtObs(*args, 0.9)
    )
    assert evalEmissivity(0.9, lam) == 0.9
