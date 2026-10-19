"""
Step Forward (SF) encoder.

The time loop has a NumPy version, which runs over every sample and feature at once, and a Numba version, which
compiles the loop and runs the samples and features in parallel. The Numba version is used when Numba is installed
(``pip install pyspikes[numba]``); both give the same spikes.
"""

import numpy as np

from .base import HAS_NUMBA, Encoder

if HAS_NUMBA:
    from numba import njit, prange

__all__ = ["StepForward"]


class StepForward(Encoder):
    """
    Step Forward (SF) encoding [3]_.

    Each feature has a baseline that starts at its first value. When the signal rises above ``baseline + threshold``
    a positive spike (``1``) is emitted and the baseline moves up by ``threshold``; when it falls below
    ``baseline - threshold`` a negative spike (``-1``) is emitted and the baseline moves down. The spikes can be
    decoded back to a staircase approximation of the signal: ``first value + threshold * cumulative sum of spikes``.

    The threshold is in the units of the data and is the only parameter. By default it is set for each feature of
    every sample to 1.5 times the mean absolute change between time steps. This factor is not taken from the paper; it
    was chosen empirically, by how well the spikes reconstruct EEG (``tests/test-data``).

    .. [3] Petro, B., Kasabov, N., & Kiss, R. M. (2020). Selection and optimization of temporal spike encoding methods
        for spiking neural networks. IEEE Transactions on Neural Networks and Learning Systems, 31(2), 358-370.

    """

    def __init__(self, data, threshold=None):
        """

        :param ndarray data:
            See :py:mod: nucube.utils.ReadCSV.get_samples()
        :param float threshold:
            Threshold in the units of the data. It can also be a list of threshold according to the numbers of input
            features. If not given, it is 1.5 times the mean absolute change between time steps, for each feature of
            every sample.
        """

        super().__init__()

        self.data = data
        self.threshold = threshold

    def get_spikes(self):
        """
        Returns Step Forward spikes.

        :rtype: ndarray
        :return:
            Spikes (``1``, ``-1`` or ``0``) with the same shape as the data. The first time step never spikes.
        """

        signal = np.ascontiguousarray(self.data, dtype=float)

        return step_forward(signal, self._get_threshold(signal))

    def _get_threshold(self, signal):
        """Returns the threshold for each feature of every sample.

        :rtype: ndarray
        :return: Threshold of shape ``(samples, features)``.
        """

        samples_size, _, features_size = signal.shape

        if self.threshold is None:
            return 1.5 * np.mean(np.abs(np.diff(signal, axis=1)), axis=1)
        if isinstance(self.threshold, (float, int)):
            return np.full((samples_size, features_size), float(self.threshold))
        if isinstance(self.threshold, list):
            return np.tile(np.asarray(self.threshold, dtype=float), (samples_size, 1))

        raise TypeError("Unknown type given.")


def step_forward_numpy(signal, threshold):
    """Runs step forward encoding over every sample and feature at once.

    The baseline starts at the first value. A positive spike is emitted when the signal rises above
    ``baseline + threshold`` and a negative spike when it falls below ``baseline - threshold``; either way the
    baseline moves by ``threshold`` towards the signal. The first time step never spikes.

    :param ndarray signal:
        Data of shape ``(samples, time, features)``.
    :param ndarray threshold:
        Threshold for each sample and feature, of shape ``(samples, features)``.
    :rtype: ndarray
    :return: Spikes (``1``, ``-1`` or ``0``) with the same shape as ``signal``.
    """

    output = np.zeros(signal.shape, dtype=np.int8)
    baseline = signal[:, 0, :].copy()

    for t in range(1, signal.shape[1]):
        value = signal[:, t, :]
        up = value > baseline + threshold
        down = ~up & (value < baseline - threshold)

        output[:, t, :] = up.astype(np.int8) - down.astype(np.int8)
        baseline = np.where(up, baseline + threshold, np.where(down, baseline - threshold, baseline))

    return output


if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def step_forward_numba(signal, threshold):
        """Numba version of :func:`step_forward_numpy`."""

        samples_size, time_size, features_size = signal.shape
        output = np.zeros(signal.shape, dtype=np.int8)

        for index in prange(samples_size * features_size):
            sample = index // features_size
            feature = index % features_size
            step = threshold[sample, feature]
            baseline = signal[sample, 0, feature]

            for t in range(1, time_size):
                value = signal[sample, t, feature]
                if value > baseline + step:
                    output[sample, t, feature] = 1
                    baseline = baseline + step
                elif value < baseline - step:
                    output[sample, t, feature] = -1
                    baseline = baseline - step

        return output

    step_forward = step_forward_numba
else:
    step_forward = step_forward_numpy
