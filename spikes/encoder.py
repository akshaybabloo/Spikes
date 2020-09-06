#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 Akshay Raj Gollahalli

"""
Spike Encoders
==============

In this module we will look at some in-built spike encoders.

There are two encoders provided with this library which  can be used as it is or by spiNNaker.

1. Temporal Based Encoding
2. Bens Spiker Algorithm
"""
import operator

import numpy as np
from scipy.signal import firwin

from .base import Encoder

__all__ = ['TBR', 'BSA']


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

        super(TBR, self).__init__()
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

        sample_feature_threshold = self._get_threshold()
        row_diff = self._get_training_element_difference()

        spikes = []
        for idx, sample in enumerate(row_diff):
            spikes.append(np.where(np.greater(sample, sample_feature_threshold[idx]), 1,
                                   np.where(np.less(sample, -sample_feature_threshold[idx]), -1,
                                            0)))

        # np.array([np.sign(sample_feature_threshold[i] - a) for i, a in enumerate(row_diff)])  # Possible answer

        return np.asarray(spikes, dtype=np.int8)

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

    def _get_training_element_difference(self):
        """Returns row-by-row difference for each sample.

        :rtype: numpy.ndarray
        :return: row_diff
        """

        train_data = self.data

        row_diff = []
        for index in range(train_data.shape[0]):
            row_diff.append(np.diff(train_data[index], axis=0))

        return np.asarray(row_diff)

    def _get_mean_sd_sample(self):
        """Returns mean and standard deviation of absolute data.

        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: mean, sd
        """

        diff_data = self._get_training_element_difference()
        abs_data = np.abs(diff_data)
        abs_data_shape_range = range(abs_data.shape[0])

        mean = []
        sd = []
        for index in abs_data_shape_range:
            mean.append(np.mean(abs_data[index], axis=0))

        for index in abs_data_shape_range:
            sd.append(np.std(abs_data[index], axis=0))

        return np.asarray(mean), np.asarray(sd)

    def _get_threshold(self):
        """Returns threshold values for each feature of every sample.

        :rtype: ndarray
        :return: threshold
        """

        mean, sd = self._get_mean_sd_sample()

        return np.add(mean, np.multiply(sd, self.threshold))


class BSA(Encoder):
    """
    Bens Spiker Algorithm (BSA) uses Finite Impulse Response (FIR) filter to try a reverse convolution of the
    stimulation (converting spikes to analog values). A principle borrowed from unpublished modified
    Hough Spiker Algorithm (HSA) [2]_.

    .. [2] Schrauwen, B., & Van Campenhout, J. (2003, July). BSA, a fast and accurate spike train encoding scheme.
        In Proceedings of the international joint conference on neural networks (Vol. 4, No. 4, pp. 2825-2830).
        Piscataway, NJ: IEEE.

    """

    def __init__(self, data, threshold=0.5, filter_length=2, cutoff=0.8):
        """

        :param ndarray data:
            See :py:mod: nucube.utils.ReadCSV.get_samples()
        :param float threshold:
            Threshold level for spike. It can also be a list of threshold according to the numbers of input features.
        :param int filter_length:
        :param float cutoff:
        """

        super(BSA, self).__init__()

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

        sample_size = self.data.shape[0]
        sample_rate = self.data.shape[1]
        features_size = self.data.shape[2]
        fir_filter = self._fir_filter()

        if isinstance(self.threshold, float) or isinstance(self.threshold, int):
            bsa_row_vector = np.ones(
                features_size) * self.threshold  # Transforming to scalar vector
        elif isinstance(self.threshold, list):
            bsa_row_vector = self.threshold
        else:
            raise TypeError("Unknown type given.")

        min_data = np.array([np.min(_sample, axis=0) for _sample in self.data])
        max_data = np.array([np.max(_sample, axis=0) for _sample in self.data])

        min_data_tile = np.array(
            [np.tile(min_data[num], (sample_rate, 1)) for num in range(sample_size)])
        max_data_tile = np.array(
            [np.tile(max_data[num], (sample_rate, 1)) for num in range(sample_size)])

        encoding_data = np.subtract(self.data, min_data_tile)
        encoding_data = np.divide(encoding_data,
                                  list(map(operator.sub, max_data_tile, min_data_tile)))

        output = np.zeros([sample_size, sample_rate, features_size], dtype=np.int8)

        # BSA Algorithm
        for _sample in range(0, sample_size):
            for _feature in range(0, features_size):
                for _idx in range(0, sample_rate - fir_filter['filter_length']):
                    error1 = 0
                    error2 = 0

                    for j in range(0, fir_filter['filter_length']):
                        error1 += abs(
                            encoding_data[_sample][_idx][_feature] - fir_filter['filter_values'][j])
                        error2 += abs(encoding_data[_sample][_idx][_feature])

                    if error1 <= (error2 - bsa_row_vector[_feature]):
                        output[_sample][_idx][_feature] = 1

                        for j in range(0, fir_filter['filter_length']):
                            if _idx + j <= features_size:
                                encoding_data[_sample][_idx][_feature] -= \
                                    fir_filter['filter_values'][j]

        return output

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

    def _fir_filter(self):
        """This method returns FIR filter using window method.

        :rtype: dict
        :return flow:
            The dictionary contains the following:

            - ``filter_values``: A list of float.
            - ``filter_length``: The length of ``filter_value``

        """
        filter_values = firwin(self.filter_length, cutoff=self.cutoff)

        flow = {'filter_values': filter_values, 'filter_length': len(filter_values)}

        return flow
