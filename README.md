# Spike Encoders

| Branch | Codecov | CI |
|--------|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Master | Soon.. | ![Python application](https://github.com/akshaybabloo/Spikes/workflows/Python%20application/badge.svg) |

Spike encoders for Spiking Neural Network.

This package consists of two types of spike encoders for spatio-temporal data:

1. Threshold Based Representation (TBR) encoder
2. Bens Spiker Algorithm (BSA) encoder

<!-- TOC -->

- [Data](#data)
- [Instillation](#instillation)
- [Example](#example)
- [Contribution](#contribution)
- [Issues](#issues)

<!-- /TOC -->

## Data

The data given to the encoders are spatio-temporal. Each sample is one `csv` file. In each file, every column is a feature and the rows are time points.

For example each file given in the [Data](https://github.com/akshaybabloo/Spikes/tree/master/Data) folder had 128 rows and 14 columns, 14 columns are the features and 128 columns are the data points.

## Instillation

```
pip install pyspikes
```

## Example

```python
from spikes import encoder
from spikes.utility import ReadCSV

data = ReadCSV('Data').get_samples()['samples']

bsa = encoder.BSA(data)
print(bsa.get_spikes())

tbr = encoder.TBR(data)
print(tbr.get_spikes())
```

## Contribution

All contributions are welcome.

## Issues

Issues can be opened through Github's [Issues](https://github.com/akshaybabloo/Spikes/issues) tab.
