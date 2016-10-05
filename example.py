#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 Akshay Raj Gollahalli

from __future__ import print_function
import os

from Spikes import BSA, TBR

data_location = os.path.abspath('Data')

print('BSA Encoder')
spikes_bsa = BSA(data_location)
print(spikes_bsa.get_spikes())

print('TBR Encoder')
spikes_tbr = TBR(data_location)
print(spikes_tbr.get_spikes())
