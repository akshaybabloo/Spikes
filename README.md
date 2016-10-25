# Spike Encoders

Spike encoders for Spiking Neural Network.

This package consists of two types of spike encoders for spatio-temporal data:

1. Threshold Based Representation (TBR) encoder
2. Bens Spiker Algorithm (BSA) encoder

## Data

The data given to the encoders are spatio-temporal. Each sample is one `csv` file. In each file, every column is a feature and the rows are time points.

For example each file given in the [Data](https://github.com/akshaybabloo/Spikes/tree/master/Data) folder had 128 rows and 14 columns, 14 columns are the features and 128 columns are the data points.

## Instillation

```
python setup.py install
```

## Contribution

All contributions are welcome.

## Issues

Issues can be opened through Github's [Issues](https://github.com/akshaybabloo/Spikes/issues) tab.

## Example

```python
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
```