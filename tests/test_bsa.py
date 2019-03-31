import numpy as np
import numpy.testing as npt
import os

from spikes.encoder import BSA
from spikes.utility import ReadCSV


def test_get_spikes():
    path = os.path.abspath('test_spikes_BSA.npy')

    test_spikes = np.load(path)
    data = ReadCSV('../Data').get_samples()['samples']

    spikes = BSA(data).get_spikes()

    npt.assert_array_equal(spikes, test_spikes)


def test_get_spikes_time():
    path = os.path.abspath('test_spikes_time_BSA.npy')

    test_spikes_time = np.load(path)
    data = ReadCSV('../Data').get_samples()['samples']

    spikes_time = BSA(data).get_spike_time()

    npt.assert_array_equal(spikes_time, test_spikes_time)
