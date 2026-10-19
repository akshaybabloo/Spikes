"""
Threshold Based Representation (TBR) encoder.
"""

import numpy as np

from .base import Encoder

__all__ = ["TBR"]


class TBR(Encoder):
    """
    Threshold Based Representation encoding [1]_

    .. [1] Delbruck, T., & Lichtsteiner, P. (2007, May). Fast sensory motor control based on event-based hybrid
       neuromorphic-procedural system. In 2007 IEEE International Symposium on Circuits and Systems (pp. 845-848). IEEE.

    """

    def __init__(self, data, threshold=0.5):
        """
        :param numpy.ndarray data:
            See :py:mod: nucube.utils.ReadCSV
        :param float threshold:
            Threshold to cut off spike detection.
        """

        super().__init__()
        self.data = data
        self.threshold = threshold

    def get_spikes(self):
        """Returns TBR spikes.

        :rtype: numpy.ndarray
        :return: spikes

        >>> spikes = TBR()
        >>> spikes.get_spikes()
            [[[0 0 0 ..., 0 0 0]
            [1 1 1 ..., 1 1 1]
            [0 0 1 ..., 1 0 1]
            ...,
            [0 0 0 ..., 1 0 0]
            [0 0 0 ..., 0 0 1]
            [0 0 0 ..., 0 0 0]]
            |
            [[0 0 0 ..., 0 0 0]
            [0 0 0 ..., 0 0 0]
            [0 1 0 ..., 0 1 1]
            ...,
            [0 0 0 ..., 0 0 0]
            [0 0 0 ..., 0 0 1]
            [1 0 1 ..., 0 0 1]]]

        """

        diff = np.diff(self.data, axis=1)
        threshold = self._get_threshold(diff)[:, np.newaxis, :]

        return (diff > threshold).astype(np.int8) - (diff < -threshold).astype(np.int8)

    def _get_threshold(self, diff):
        """Returns the threshold for each feature of every sample: the mean absolute change between time steps plus
        ``threshold`` times its standard deviation.

        :param ndarray diff:
            Change between time steps, of shape ``(samples, time - 1, features)``.
        :rtype: ndarray
        :return: Threshold of shape ``(samples, features)``.
        """

        abs_diff = np.abs(diff)

        return np.mean(abs_diff, axis=1) + self.threshold * np.std(abs_diff, axis=1)
