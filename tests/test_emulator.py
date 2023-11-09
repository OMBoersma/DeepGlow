import pytest
from DeepGlow import Emulator
import numpy as np


# Define a fixture that yields the Emulator instance.
@pytest.fixture
def emulator():
    return Emulator()


# Assert correct initialisation of TF model.
def test_initialization(emulator):
    assert emulator.NNmodel is not None


def test_flux_calculation(emulator):
    GRB_params = np.array([0, -1, 0, 0, 0.1, 0.1, 2.2, -2, -2, 0])
    observing_times = np.array([1e5, 1e6, 1e7])
    observing_frequencies = np.array([1e9, 1e12, 1e15])

    flux = emulator.flux(
        params=GRB_params, t_obs=observing_times, nu_obs=observing_frequencies
    )
    expected_flux = np.array([5.75068180e-01, 8.58790301e-01, 5.39014321e-05])

    assert len(flux) == len(observing_times)
    np.testing.assert_almost_equal(flux, expected_flux, decimal=3)
