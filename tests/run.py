from __future__ import print_function
import os
import numpy as np
import numpy.testing as npt

from Spikes.Utility import ReadCSV
from Spikes.Encoder import BSA, TBR

data_location = os.path.abspath('Data')


def test_BSA():
    load_data = np.load('test_data_BSA.npy')

    read_data_for_BSA = ReadCSV(data_location)
    spikes_bsa = BSA(read_data_for_BSA)

    npt.assert_array_equal(load_data, spikes_bsa.get_spikes())


def test_TBR():
    load_data = np.load('test_data_TBR.npy')

    read_data_for_TBR = ReadCSV(data_location)
    spikes_tbr = TBR(read_data_for_TBR.read_samples())

    npt.assert_array_equal(load_data, spikes_tbr.get_spikes())
