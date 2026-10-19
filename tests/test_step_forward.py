import os

import numpy as np
import numpy.testing as npt
import pytest

from spikes import _kernels
from spikes.encoder import StepForward
from spikes.utility import ReadCSV


def test_get_spikes():
    test_spikes = np.load(os.path.join("tests", "test_spikes_StepForward.npy"))
    data = ReadCSV(os.path.join("tests", "test-data")).get_samples()["samples"]

    spikes = StepForward(data).get_spikes()

    npt.assert_array_equal(spikes, test_spikes)


def test_get_spikes_time():
    test_spikes_time = np.load(os.path.join("tests", "test_spikes_time_StepForward-1.npy"), allow_pickle=True)
    data = ReadCSV(os.path.join("tests", "test-data")).get_samples()["samples"]

    spikes_time = StepForward(data).get_spike_time()

    assert spikes_time == test_spikes_time.tolist()


def test_step_forward_follows_baseline(backend):
    # Feature 0 steps up twice, holds, then steps down; feature 1's threshold is too high to spike at all.
    feature = [0.0, 0.4, 1.0, 1.2, 0.5, 0.0]
    signal = np.array([feature, feature]).T[np.newaxis].copy()

    spikes = getattr(_kernels, f"step_forward_{backend}")(signal, np.array([[0.5, 2.0]]))

    npt.assert_array_equal(spikes[0, :, 0], [0, 0, 1, 1, 0, -1])
    npt.assert_array_equal(spikes[0, :, 1], [0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize(
    "threshold, expected",
    [(0.5, [[0.5, 0.5]]), ([0.5, 2.0], [[0.5, 2.0]]), (None, [[1.5 * 0.4, 1.5 * 0.2]])],
)
def test_threshold(threshold, expected):
    data = np.array([[[0.0, 0.0], [0.4, 0.2], [0.0, 0.0], [0.4, 0.2]]])

    npt.assert_allclose(StepForward(data, threshold)._get_threshold(data), expected)


def test_unknown_threshold_type():
    with pytest.raises(TypeError):
        StepForward(np.zeros((1, 4, 1)), threshold="0.5").get_spikes()
