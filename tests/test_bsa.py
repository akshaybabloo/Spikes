import os

import numpy as np
import numpy.testing as npt

from spikes.encoder import BSA
from spikes.utility import ReadCSV


def test_get_spikes():
    test_spikes = np.load(os.path.join("tests", "test_spikes_BSA.npy"))
    data = ReadCSV("./Data").get_samples()["samples"]

    spikes = BSA(data).get_spikes()

    npt.assert_array_equal(spikes, test_spikes)


def test_get_spikes_time():
    test_spikes_time = np.load(os.path.join("tests", "test_spikes_time_BSA-1.npy"), allow_pickle=True)
    data = ReadCSV("./Data").get_samples()["samples"]

    spikes_time = BSA(data).get_spike_time()

    assert spikes_time == test_spikes_time.tolist()


def test_bsa_spikes_where_filter_fits():
    # Feature 0 fires once the window matches the filter; feature 1's threshold is too high to fire at all.
    signal = np.array([[[0.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]])
    filter_values = np.array([0.5, 0.5])

    spikes = BSA._bsa(signal, filter_values, np.array([0.1, 2.0]))

    npt.assert_array_equal(spikes[0], [[0, 0], [1, 0], [0, 0], [0, 0]])
    npt.assert_array_equal(signal[0, :, 0], [0.0, 0.0, 0.0, 0.0])


def test_bsa_spikes_at_end_of_signal():
    signal = np.array([[[0.0], [0.0], [0.0], [0.5]]])

    spikes = BSA._bsa(signal, np.array([0.5, 0.5]), np.array([0.1]))

    npt.assert_array_equal(spikes[0, :, 0], [0, 0, 0, 1])
