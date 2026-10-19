# Spike Encoders

| Branch | Codecov | CI |
|--------|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Master | [![codecov](https://codecov.io/gh/akshaybabloo/Spikes/graph/badge.svg?token=twI3KZYyLu)](https://codecov.io/gh/akshaybabloo/Spikes) | ![Python application](https://github.com/akshaybabloo/Spikes/workflows/Python%20application/badge.svg) |

Spike encoders for Spiking Neural Network.

This package consists of three types of spike encoders for spatio-temporal data:

1. Threshold Based Representation (TBR) encoder
2. Bens Spiker Algorithm (BSA) encoder
3. Step Forward (SF) encoder

<!-- TOC -->

- [Data](#data)
- [Instillation](#instillation)
- [Example](#example)
- [Contribution](#contribution)
- [Issues](#issues)

<!-- /TOC -->

## Data

The data given to the encoders are spatio-temporal. Each sample is one `csv` file. In each file, every column is a feature and the rows are time points.

For example, each file in [tests/test-data](https://github.com/akshaybabloo/Spikes/tree/master/tests/test-data) is one second of EEG from the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/): 160 rows (time points at 160 Hz) and 14 columns (EEG channels). See its [README](tests/test-data/README.md) for details and citation.

## Instillation

```
pip install pyspikes
```

BSA and Step Forward run faster with [Numba](https://numba.pydata.org/) installed. The spikes are the same either way:

```
pip install pyspikes[numba]
```

## Example

```python
from spikes import encoder
from spikes.utility import ReadCSV

data = ReadCSV("tests/test-data").get_samples()["samples"]

bsa = encoder.BSA(data)
print(bsa.get_spikes())

tbr = encoder.TBR(data)
print(tbr.get_spikes())

sf = encoder.StepForward(data)
print(sf.get_spikes())
```

## Contribution

All contributions are welcome.

## Issues

Issues can be opened through Github's [Issues](https://github.com/akshaybabloo/Spikes/issues) tab.
