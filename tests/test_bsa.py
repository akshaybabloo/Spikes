import os

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import firwin

from spikes.encoder import BSA, bsa
from spikes.encoder.base import HAS_NUMBA
from spikes.utility import ReadCSV


def test_get_spikes():
    test_spikes = np.load(os.path.join("tests", "test_spikes_BSA.npy"))
    data = ReadCSV(os.path.join("tests", "test-data")).get_samples()["samples"]

    spikes = BSA(data).get_spikes()

    npt.assert_array_equal(spikes, test_spikes)


def test_get_spikes_time():
    test_spikes_time = np.load(os.path.join("tests", "test_spikes_time_BSA-1.npy"), allow_pickle=True)
    data = ReadCSV(os.path.join("tests", "test-data")).get_samples()["samples"]

    spikes_time = BSA(data).get_spike_time()

    assert spikes_time == test_spikes_time.tolist()


def test_bsa_spikes_where_filter_fits(backend):
    # Feature 0 fires once the window matches the filter; feature 1's threshold is too high to fire at all.
    signal = np.array([[[0.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]])
    filter_values = np.array([0.5, 0.5])

    spikes = getattr(bsa, f"bsa_{backend}")(signal, filter_values, np.array([0.1, 2.0]))

    npt.assert_array_equal(spikes[0], [[0, 0], [1, 0], [0, 0], [0, 0]])
    npt.assert_array_equal(signal[0, :, 0], [0.0, 0.0, 0.0, 0.0])


def test_bsa_spikes_at_end_of_signal(backend):
    signal = np.array([[[0.0], [0.0], [0.0], [0.5]]])

    spikes = getattr(bsa, f"bsa_{backend}")(signal, np.array([0.5, 0.5]), np.array([0.1]))

    npt.assert_array_equal(spikes[0, :, 0], [0, 0, 0, 1])


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba is not installed")
@pytest.mark.parametrize("filter_length, cutoff", [(2, 0.8), (12, 0.2), (30, 0.05)])
def test_backends_match(random_walks, filter_length, cutoff):
    signal = (random_walks - random_walks.min(axis=1, keepdims=True)) / np.ptp(random_walks, axis=1, keepdims=True)
    filter_values = firwin(filter_length, cutoff=cutoff)
    threshold = np.random.default_rng(7).uniform(0, 1, size=random_walks.shape[2])

    expected = bsa.bsa_numpy(signal.copy(), filter_values, threshold)

    npt.assert_array_equal(bsa.bsa_numba(signal.copy(), filter_values, threshold), expected)
