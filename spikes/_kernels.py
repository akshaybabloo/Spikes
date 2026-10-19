"""
Time loops of the encoders.

Each loop has a NumPy version, which runs over every sample and feature at once, and a Numba version, which compiles
the loop and runs the samples and features in parallel. The Numba versions are used when Numba is installed
(``pip install pyspikes[numba]``); both give the same spikes.

Only Step Forward uses these so far.
"""

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    HAS_NUMBA = False
else:
    HAS_NUMBA = True


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
