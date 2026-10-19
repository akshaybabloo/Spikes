"""
Bens Spiker Algorithm (BSA) encoder.

The time loop has a NumPy version, which runs over every sample and feature at once, and a Numba version, which
compiles the loop and runs the samples and features in parallel. The Numba version is used when Numba is installed
(``pip install pyspikes[numba]``); both give the same spikes.
"""

import numpy as np
from scipy.signal import firwin

from .base import HAS_NUMBA, Encoder

if HAS_NUMBA:
    from numba import njit, prange

__all__ = ["BSA"]


class BSA(Encoder):
    """
    Bens Spiker Algorithm (BSA) uses Finite Impulse Response (FIR) filter to try a reverse convolution of the
    stimulation (converting spikes to analog values). A principle borrowed from unpublished modified
    Hough Spiker Algorithm (HSA) [2]_.

    The defaults for ``threshold``, ``filter_length`` and ``cutoff`` are not taken from the paper; the best values
    depend on the signal. They were chosen empirically, by how well the spikes reconstruct EEG (``tests/test-data``)
    and smooth synthetic signals. For other data, tune them by reconstruction error, for example the signal-to-noise
    ratio between the normalised input and the spikes convolved with the filter.

    .. [2] Schrauwen, B., & Van Campenhout, J. (2003, July). BSA, a fast and accurate spike train encoding scheme.
        In Proceedings of the international joint conference on neural networks (Vol. 4, No. 4, pp. 2825-2830).
        Piscataway, NJ: IEEE.

    """

    def __init__(self, data, threshold=0.75, filter_length=12, cutoff=0.2):
        """

        :param ndarray data:
            See :py:mod: nucube.utils.ReadCSV.get_samples()
        :param float threshold:
            Threshold level for spike. It can also be a list of threshold according to the numbers of input features.
            It is subtracted, as in the paper: a spike fires when ``error1 <= error2 - threshold``. Some other
            implementations multiply instead (``error2 * threshold``), so their thresholds are not comparable.
        :param int filter_length:
            Number of taps in the FIR filter. Longer filters follow slower changes in the signal.
        :param float cutoff:
            Cutoff frequency of the FIR filter, as a fraction of the Nyquist frequency.
        """

        super().__init__()

        self.data = data
        self.threshold = threshold
        self.filter_length = filter_length
        self.cutoff = cutoff

    def get_spikes(self):
        """
        Returns BSA spikes.

        :rtype: ndarray
        :return output:
            Returns an multidimensional ndarray where each ndarray of ndarray is a sample.

        See ReadData on how to setup the files.

        >>> spike_object = BSA()
        >>> spike_object.get_spikes()
        [[[1 1 1 ... 0 0 0]
          [1 1 1 ... 1 1 1]
          [1 1 1 ... 0 1 1]
          ...
          [1 1 1 ... 1 1 1]
          [0 0 0 ... 0 0 0]
          [0 0 0 ... 0 0 0]]
         [[0 1 1 ... 1 1 1]
          [1 1 1 ... 1 1 1]
          [0 1 0 ... 1 0 1]
          ...
          [0 0 0 ... 0 0 0]
          [0 0 0 ... 0 0 0]
          [0 0 0 ... 0 0 0]]]
        """

        features_size = self.data.shape[2]
        fir_filter = self._fir_filter()

        if isinstance(self.threshold, (float, int)):
            bsa_row_vector = np.ones(features_size) * self.threshold  # Transforming to scalar vector
        elif isinstance(self.threshold, list):
            bsa_row_vector = np.asarray(self.threshold, dtype=float)
        else:
            raise TypeError("Unknown type given.")

        # Normalise every feature of every sample to [0, 1]
        min_data = np.min(self.data, axis=1, keepdims=True)
        max_data = np.max(self.data, axis=1, keepdims=True)
        encoding_data = (self.data - min_data) / (max_data - min_data)

        return bsa(encoding_data, fir_filter["filter_values"], bsa_row_vector)

    def _fir_filter(self):
        """This method returns FIR filter using window method.

        :rtype: dict
        :return flow:
            The dictionary contains the following:

            - ``filter_values``: A list of float.
            - ``filter_length``: The length of ``filter_value``

        """
        filter_values = firwin(self.filter_length, cutoff=self.cutoff)

        flow = {"filter_values": filter_values, "filter_length": len(filter_values)}

        return flow


def bsa_numpy(signal, filter_values, threshold):
    """Runs BSA over every sample and feature at once.

    Follows the pseudo-code in Schrauwen & Van Campenhout (2003): at time ``t`` the filter is compared with
    ``signal[t:t + len(filter)]``, and if it fits a spike is emitted and the filter is subtracted from that window.
    Near the end of the signal the window, and the filter, are cut short.

    :param ndarray signal:
        Normalised data of shape ``(samples, time, features)``. It is modified in place.
    :param ndarray filter_values:
        FIR filter taps.
    :param ndarray threshold:
        Threshold for each feature.
    :rtype: ndarray
    :return: Spikes with the same shape as ``signal``.
    """

    time_size = signal.shape[1]
    output = np.zeros(signal.shape, dtype=np.int8)

    for t in range(time_size):
        window = signal[:, t : t + len(filter_values), :]
        taps = filter_values[: window.shape[1], np.newaxis]

        error1 = np.sum(np.abs(window - taps), axis=1)
        error2 = np.sum(np.abs(window), axis=1)
        fire = error1 <= (error2 - threshold)

        output[:, t, :] = fire
        window -= fire[:, np.newaxis, :] * taps

    return output


if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def bsa_numba(signal, filter_values, threshold):
        """Numba version of :func:`bsa_numpy`."""

        samples_size, time_size, features_size = signal.shape
        output = np.zeros(signal.shape, dtype=np.int8)

        for index in prange(samples_size * features_size):
            sample = index // features_size
            feature = index % features_size

            for t in range(time_size):
                width = min(len(filter_values), time_size - t)
                error1 = 0.0
                error2 = 0.0
                for j in range(width):
                    error1 += abs(signal[sample, t + j, feature] - filter_values[j])
                    error2 += abs(signal[sample, t + j, feature])

                if error1 <= error2 - threshold[feature]:
                    output[sample, t, feature] = 1
                    for j in range(width):
                        signal[sample, t + j, feature] -= filter_values[j]

        return output

    bsa = bsa_numba
else:
    bsa = bsa_numpy
