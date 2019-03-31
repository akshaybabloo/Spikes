import numpy as np
import numpy.testing as npt
import os

from spikes.encoder import TBR
from spikes.utility import ReadCSV


def test_get_spikes():
    path = os.path.abspath('test_spikes_TBR.npy')

    test_spikes = np.load(path)
    data = ReadCSV('../Data').get_samples()['samples']

    spikes = TBR(data).get_spikes()

    npt.assert_array_equal(spikes, test_spikes)


def test_get_spikes_time():
    path = os.path.abspath('test_spikes_time_TBR.npy')

    test_spikes_time = np.load(path)
    data = ReadCSV('../Data').get_samples()['samples']

    spikes_time = TBR(data).get_spike_time()

    npt.assert_array_equal(spikes_time, test_spikes_time)
