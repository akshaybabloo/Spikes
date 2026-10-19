"""
Spike Encoders
==============

In this module we will look at some in-built spike encoders.

There are three encoders provided with this library which  can be used as it is or by spiNNaker.

1. Temporal Based Encoding
2. Bens Spiker Algorithm
3. Step Forward
"""

import numpy as np
from scipy.signal import firwin

from . import _kernels
from .base import Encoder

__all__ = ["BSA", "TBR", "StepForward"]


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

        return _kernels.bsa(encoding_data, fir_filter["filter_values"], bsa_row_vector)

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

        return _kernels.step_forward(signal, self._get_threshold(signal))

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
