#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 Akshay Raj Gollahalli
"""
Reads the data from folders and json files.
"""

import csv
import logging
import os
import re
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from .errors import LengthMismatchError, NotEnoughDataError, SplitDataError
from .utils import log_it

__all__ = ['ReadCSV']


class ReadCSV(object):
    """Reads the CSV files in a given folder.

    If no folder names is give, this program searches for folder ``Data``; If
    no location is given.

    :param str path:
        Location of the directory containing the data files.
    :param bool log: Default False.
        Verbose mode.
    """

    def __init__(self, path, log=False):

        self.yes_no = log
        self.data_folder = os.path.abspath(path) + os.sep

        log_it(self.yes_no)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("ReadCSV class called")

        try:
            if os.path.isdir(self.data_folder):
                self.prefixed = [filename for filename in os.listdir(self.data_folder) if
                                 filename.startswith("sam")]
                self.logger.debug("Reading the data")
            else:
                self.logger.debug("%s does not exist", self.data_folder)
        except IOError as err:
            self.logger.exception("Data files not found - %s", err)
            sys.exit(1)

        if len(self.prefixed) == 1:
            raise NotEnoughDataError("There should be more than one sample to continue.")

    def get_samples(self):
        """
        Reads all the sample starting with sam*.csv and returns a Pandas DataFrame.

        :rtype: dict
        :return samples:
            A dictionary of ``samples`` and ``labels`` (this depends if you have labels file).

        >>> data = ReadCSV('Data')
        >>> data.get_samples()
            {
            'labels': [...],
            'samples: array(...)
            }
        """
        self.prefixed.sort(key=natural_keys)  # Sorted with filename and sample number

        abs_path_of_samples = [self.data_folder + self.prefixed for self.prefixed in self.prefixed]
        raw_samples_list = np.asarray(
            [np.genfromtxt(sample, delimiter=',') for sample in abs_path_of_samples])

        labels = self._get_class_labels()

        if labels[0] is not False:
            if len(raw_samples_list) != len(labels):
                LengthMismatchError(
                    "Number of samples ({0}) is not equal to number of labels ({1}).".format(
                        len(raw_samples_list),
                        len(labels)))
            samples = {'samples': raw_samples_list, 'labels': labels}
        else:
            samples = {'samples': raw_samples_list}

        return samples

    def get_split_data(self, split_to=0.5):
        """Split samples into training and testing data.

        The default test size is 0.5 (50%)

        :param float split_to:
            Percentage split training and testing. Defaults to ``0.5``.

        :rtype: dict
        :return:
            Dictionary of ``train_samples``, ``train_labels``, ``test_samples``,
            ``test_labels`` and ``split_percentage``.
        """

        samples = self.get_samples()

        if split_to > 1:
            raise SplitDataError("Split data should be less that 1.0")

        train_samples, train_labels, test_samples, test_labels = train_test_split(
            samples['samples'],
            samples['labels'],
            test_size=split_to)

        self.logger.debug("Data has been split into Training and Testing data")

        return {'train_samples': train_samples, 'train_labels': train_labels,
                'test_samples': test_samples,
                'test_labels': test_labels,
                'split_percentage': split_to}

    def sample_size(self):
        """Returns the length of the sample size.

        :rtype: int
        :return size:
            Length on samples.
        """
        size = len(self.prefixed)
        self.logger.debug("Sample size returned")
        return size

    def time_feature_length(self):
        """
        Returns the time length of a file by counting it's number of columns.

        :rtype: dict
        :return data_dict:
            Dictionary of ``time_length`` and ``feature_length``.
        """
        _file = self.prefixed[0]
        sample = []
        with open(self.data_folder + _file) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for col in reader:
                sample.append(col)

        data_dict = {'time_length': len(sample), 'feature_length': len(sample[0][0].split(','))}
        self.logger.debug("Feature length returned")
        return data_dict

    def _get_class_labels(self):
        _labels = []
        if os.path.isfile(self.data_folder + 'tar_class_labels.csv'):
            with open(self.data_folder + 'tar_class_labels.csv', 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
                for row in reader:
                    _labels.append(int(row[0]))
        else:
            _labels.append(False)

        self.logger.debug("Class labels returned")
        return _labels


def atoi(text):
    """Checks if the file names contain numbers.

    :param str text:
        File name.

    :rtype: bool
    :return: True or false based on the digits in text.
    """
    logging.debug("File names checked")
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Splits the number from the file name.

    :param str text:
        File name.
    :rtype: list
    :return:
        Splits the ``text`` if number is present in it.
    """
    logging.debug("File names split")
    return [atoi(num) for num in re.split(r'(\d+)', text)]
