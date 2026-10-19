"""
Base file for all classes.
"""

import numpy as np


class Encoder:
    """
    Base for encoders.
    """

    def __init__(self):
        pass

    def get_spikes(self):
        pass

    def get_spike_time(self, offset=100):
        """
        Merges the samples and returns the time at which the spike happens. Spike time always startes with ``1`` and
        increases with :param: offset.

        :param int offset:
            Shifting the spike index to the given number, if the first index is ``0``, then it will always be ``1``
        :rtype: list
        :return: Spike time with an offset.
        """

        reshaped_raw_spikes = np.vstack(self.get_spikes())

        spike_times = []

        for _feature in reshaped_raw_spikes.T:
            _idx_of_ones = np.squeeze(np.add(np.where(_feature == 1), 1))
            if _idx_of_ones[0] == 1:
                spike_times.append(np.insert(np.multiply(_idx_of_ones[1:], offset), 0, 1).tolist())
            else:
                spike_times.append(np.multiply(_idx_of_ones, offset).tolist())

        return spike_times

    def save(self):
        pass
