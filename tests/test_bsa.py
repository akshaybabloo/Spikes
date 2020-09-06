import numpy as np
import numpy.testing as npt

from spikes.encoder import BSA
from spikes.utility import ReadCSV


def test_get_spikes():
    test_spikes = np.load('./test_spikes_BSA.npy')
    data = ReadCSV('../Data').get_samples()['samples']

    spikes = BSA(data).get_spikes()

    npt.assert_array_equal(spikes, test_spikes)


def test_get_spikes_time():
    test_spikes_time = np.load('./test_spikes_time_BSA.npy')
    data = ReadCSV('../Data').get_samples()['samples']

    spikes_time = BSA(data).get_spike_time()

    npt.assert_array_equal(spikes_time, test_spikes_time)
