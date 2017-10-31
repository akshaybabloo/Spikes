#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 Akshay Raj Gollahalli

import operator

import numpy as np
import pandas as pd
from scipy.signal import firwin

from .utility import ReadCSV

__all__ = ['BSA', 'TBR']


class BSA:
    """
    Bens Spiker Algorithm (BSA) uses Finite Impulse Response (FIR) filter to try a reverse convolution of the
    stimulation (converting spikes to analog values). A principle borrowed from unpublished modified
    Hough Spiker Algorithm (HSA).

    References
    ----------
    .. [1] Schrauwen, B., & Van Campenhout, J. (2003, July). BSA, a fast and accurate spike train encoding scheme.
        In Proceedings of the international joint conference on neural networks (Vol. 4, No. 4, pp. 2825-2830).
        Piscataway, NJ: IEEE.

    Note
    ----

    This code is not vetted.
    """

    def __init__(self, data, threshold=0.5):
        """

        Parameters
        ----------
        data
            Absolute Location of samples (folder location).
        threshold
            Threshold level for spike. It can also be a list of threshold according to the numbers of input features.
        """

        self.threshold = threshold
        self.file_reader = data

    def get_spikes(self):
        """
        Returns BSA spikes.

        Returns
        -------

        output  :  ndarray
            Returns an multidimensional ndarray where each ndarray of ndarray is a sample.

        Example
        -------

        See ReadData on how to setup the files.

        >>> spike_object = BSA()
        >>> spike_object.get_spikes()
        [[[ 0.  1.  1. ...,  0.  0.  0.]
        [ 1.  1.  1. ...,  0.  1.  1.]
        [ 1.  1.  1. ...,  0.  1.  1.]
        ...,
        [ 1.  1.  1. ...,  0.  1.  0.]
        [ 1.  1.  1. ...,  0.  1.  1.]
        [ 0.  0.  0. ...,  0.  0.  0.]]
        ...
        [[ 0.  1.  0. ...,  1.  1.  1.]
        [ 0.  1.  0. ...,  1.  1.  1.]
        [ 0.  0.  0. ...,  1.  0.  0.]
        ...,
        [ 0.  0.  0. ...,  0.  0.  0.]
        [ 0.  0.  0. ...,  0.  0.  0.]
        [ 0.  0.  0. ...,  0.  0.  0.]]]
        """
        features = self.file_reader.time_feature_length()
        sample_number = self.file_reader.sample_size()
        data = self.file_reader.read_samples()
        fir_filter = self.fir_filter()

        if type(self.threshold).__name__ == "float" or type(self.threshold).__name__ == "int":
            bsa_row_vector = np.ones(features['feature_length']) * self.threshold  # Transforming to scalar vector
        elif type(self.threshold).__name__ == "list":
            bsa_row_vector = self.threshold
        else:
            raise TypeError

        min_data = [df.min(axis=0).to_frame().T for df in data[0]]
        max_data = [df.max(axis=0).to_frame().T for df in data[0]]

        min_data_tile = [np.tile(min_data[num], (features['time_length'], 1)) for num in range(sample_number)]
        max_data_tile = [np.tile(max_data[num], (features['time_length'], 1)) for num in range(sample_number)]

        encoding_data = data[0].subtract(min_data_tile)
        encoding_data = encoding_data.divide(list(map(operator.sub, max_data_tile, min_data_tile)))

        output = np.zeros([sample_number, features['time_length'], features['feature_length']])

        # BSA Algorithm
        for s in range(0, sample_number):
            for f in range(0, features['feature_length']):
                for i in range(0, features['time_length'] - fir_filter['filter_length'] + 1):
                    error1 = 0
                    error2 = 0
                    for j in range(0, fir_filter['filter_length']):
                        error1 += abs(encoding_data.iloc[s][f][i] - fir_filter['filter_values'][j])
                        error2 += abs(encoding_data.iloc[s][f][i])

                    if error1 <= (error2 - bsa_row_vector[f]):
                        output[s][i][f] = 1

                        for j in range(0, fir_filter['filter_length']):
                            if i + j <= features['feature_length']:
                                encoding_data.iloc[s][f][i] -= fir_filter['filter_values'][j]

        return output

    @staticmethod
    def fir_filter(filter_length=2, cutoff=0.8):
        """This method returns FIR filter using window method.

        Parameters
        ----------
        filter_length  :  int
            Length of the filter.
        cutoff  :  float
            Cutoff value.

        Returns
        -------
        flow  :  dict
            The dictionary contains the following:

            - filter_values: A list of float.
            - filter_length: The length of ``filter_value``

        Examples
        --------

        >>> bsa = BSA()
        >>> bsa.fir_filter()
        {'filter_values': array([ 0.5,  0.5]), 'filter_length': 2}
        """
        filter_values = firwin(filter_length, cutoff=cutoff)

        flow = {'filter_values': filter_values, 'filter_length': len(filter_values)}

        return flow


class TBR:
    """
    Threshold Based Representation encoding.

    References
    ----------

    .. [2] Delbruck, T., & Lichtsteiner, P. (2007, May). Fast sensory motor control based on event-based hybrid
       neuromorphic-procedural system. In 2007 IEEE International Symposium on Circuits and Systems (pp. 845-848). IEEE.

    """

    def __init__(self, data, threshold=0.5):
        """

        Parameters
        ----------
        data
            Location of samples (folder location).
        threshold
            Threshold level for spike.
        """

        self.data = data
        self.threshold = threshold

    def get_spikes(self):
        """Returns TBR spikes.

        Examples
        --------

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

        Returns
        -------
        flow  :  ndarray
            A an ndarray of spikes.
        """

        # data = self.data['full_data']

        df1 = self._get_threshold()

        index = self.data.index  # TODO: There seems to be an error if the class values are not equal.

        temp1 = []
        for l in index:
            temp1.append(self.data[0][l].diff())

        df = pd.DataFrame(temp1)
        df.reset_index(inplace=True, drop=True)

        flow = []

        for l in df.index:
            flow.append(
                np.where(df[0][l].gt(df1.squeeze(), axis=1), 1, np.where(df[0][l].lt(-(df1.squeeze()), axis=1), 1, 0)))

        return np.array(flow)

    def _get_training_element_difference(self):

        train_data = self.data

        temp1 = train_data.index

        temp2 = []
        for l in temp1:
            temp2.append(train_data[0][l].diff())

        flow = pd.DataFrame(temp2, index=temp1)
        return flow

    def _get_mean_sd_sample(self):

        data1 = self._get_training_element_difference()
        data = data1.abs()

        index = data.index

        temp1 = []
        temp2 = []
        for l in index:
            temp1.append(data[0][l].mean(axis=0))

        for l in index:
            temp2.append(data[0][l].std(axis=0))

        mean1 = pd.DataFrame(temp1, index=index)
        sd1 = pd.DataFrame(temp2, index=index)

        mean2 = pd.DataFrame(mean1.mean(axis=0))
        sd2 = pd.DataFrame(sd1.std(axis=0))

        return mean2, sd2

    def _get_threshold(self):

        mean, sd = self._get_mean_sd_sample()
        temp = pd.DataFrame.multiply(sd, self.threshold)
        threshold = pd.DataFrame.add(mean, temp)

        return threshold