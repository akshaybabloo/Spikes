#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 Akshay Raj Gollahalli

from __future__ import print_function

import csv
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ['ReadCSV', 'ReadJSON']


class ReadCSV:
    """Read CSV files.

    This class reads in the CSV files starting with sam_*.csv, where * is the number.
    """
    def __init__(self, data):
        """

        Parameters
        ----------
        data
            Location of samples (folder location).

        Note
        -----

        If you are manually giving the location of the folder to ``data_folder`` option, you can ignore ``location`` option.
        """

        self.data_folder = data + os.sep
        if os.path.isdir(self.data_folder):
            self.prefixed = [filename for filename in os.listdir(self.data_folder) if filename.startswith("sam")]
        else:
            raise IOError('Data files not found')

        if len(self.prefixed) is 1:
            raise NotEnoughDataError("There should be more than one sample to continue.")

    def read_samples(self):
        """Read samples

        This method reads the files and indexes according to their sam_* number.

        Examples
        --------

        >>> files = ReadCSV()
        >>> files.read_samples()

        Returns
        -------
        flow  :  DataFrame
        """
        self.prefixed.sort(key=natural_keys)  # Sorted with filename and sample number

        temp = [self.data_folder + self.prefixed for self.prefixed in self.prefixed]
        data = [pd.read_csv(f, index_col=None, header=None) for f in temp]

        flow = pd.DataFrame(data)

        temp2 = self._get_class_labels()

        flow = flow.set_index([flow.index, temp2])
        flow.index = flow.index.set_names('index', level=0)
        flow.index = flow.index.set_names('class', level=1)
        pd.set_option('display.expand_frame_repr', False)

        return flow

    def get_split_data(self, split_to=0.5):
        """Get split data for training and testing.

        Parameters
        ----------

        split_to
            Percentage split of data.

        Returns
        -------
        flow  :  dict
            Returns a dictionary of two``DataFrame``, and one ``float``.

        Examples
        --------

        >>> files = ReadCSV()
        >>> files.get_split_data()
        {'test_data': DataFrame,
         'train_data': DataFrame,
         'full_data': DataFrame,
         'training_split': float}
        """
        data = self.read_samples()

        if split_to > 1:
            raise SplitDataException("Split data should be less that 1.0")

        train, test = train_test_split(data, test_size=split_to)

        flow = {'test_data': test, 'train_data': train, 'full_data': data, 'training_split': split_to}

        return flow

    def sample_size(self):
        """
        Returns the length of the sample size.

        Returns
        -------

        flow  :  int
            Size of the list of sample names.
        """
        flow = len(self.prefixed)
        return flow

    def time_feature_length(self):
        """
        Returns the time length of a file by counting it's number of columns.

        Returns
        -------

        flow  :  dict
            Returns a dictionary with the following:

            - ``time_length`` - Time length of a sample as *int*.
            - ``feature_length`` - Feature length of a sample as *int*
        """
        file = self.prefixed[0]
        a = []
        with open(self.data_folder + file) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for col in reader:
                a.append(col)
        csv_file.close()

        flow = {'time_length': len(a), 'feature_length': len(a[0][0].split(','))}
        return flow

    def get_feature_names(self):
        """
        Reads feature names if present

        Returns
        -------

        flow  :  dict
            The dictionary contains the following:

            - ``number_of_features`` - Number of features as *int*
            - ``name_features`` - Feature names as *list*
        """

        names = []
        number_of_features = 0
        if os.path.isfile(self.data_folder + 'feature_names_eeg.txt'):
            try:
                with open(self.data_folder + 'feature_names_eeg.txt', 'r') as f:
                    data = f.read()
                f.close()
                names = data.split('\n')
                number_of_features = len(names)
            except IOError as e:
                print("file not found - ", e)
        else:
            with open(self.data_folder + self.prefixed[0]) as f:
                number_of_features = len(f.readline().split(','))
                for x in range(1, number_of_features + 1):
                    names.append("feature {}".format(x))
                f.close()

        flow = {'number_of_features': number_of_features, 'name_features': names}
        return flow

    def _get_class_labels(self):
        """
        Gets all the target class labels.

        Returns
        -------

        flow  :  list
            List of feature names if given or it is self generated.
        """
        flow = []
        if os.path.isfile(self.data_folder + 'tar_class_labels.csv'):
            with open(self.data_folder + 'tar_class_labels.csv', 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
                for row in reader:
                    flow.append(int(row[0]))
                csv_file.close()
        else:
            for temp in range(len(self.prefixed)):
                flow.append(1)

        return flow


def atoi(text):
    """
    Checks if the file names contain numbers.

    Parameters
    ----------
    text
        This parameter could be a str or int.

    Returns
    -------

    flow  :  int, str
    """
    flow = int(text) if text.isdigit() else text
    return flow


def natural_keys(text):
    """
    Splits the number from the file name.

    Parameters
    ----------
    text
        A str parameter with number should be give, so the this method could split the contents.

    Returns
    -------
    flow  :  list
        List of strings.
    """
    flow = [atoi(c) for c in re.split('(\d+)', text)]
    return flow


####################################################################
#                                                                  #
#                           Exceptions                             #
#                                                                  #
####################################################################
class NotEnoughDataError(Exception):
    def __init__(self, message, errors=None):
        super(NotEnoughDataError, self).__init__(message)

        self.errors = errors


class SplitDataException(Exception):
    def __init__(self, message, errors=None):
        super(SplitDataException, self).__init__(message)

        self.errors = errors
