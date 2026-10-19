"""Encoder tests on synthetic signals whose expected behaviour is known in advance."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import chirp, lfilter, square

from spikes.encoder import BSA, TBR, StepForward

T = 500
t = np.arange(T)
rng = np.random.default_rng(42)

SIGNALS = {
    "sine": np.sin(2 * np.pi * t / 100),
    "two sines": np.sin(2 * np.pi * t / 120) + 0.5 * np.sin(2 * np.pi * t / 37),
    "chirp": chirp(t, f0=0.002, t1=T, f1=0.03),
    "step": (t >= T // 2).astype(float),
    "square": square(2 * np.pi * t / 125),
    "noisy sine": np.sin(2 * np.pi * t / 100) + rng.normal(0, 0.1, T),
}


def as_data(*signals):
    """Stacks 1-D signals into data of shape ``(1, time, features)``."""
    return np.stack(signals, axis=1)[np.newaxis]


def bsa_snr_db(signal):
    """Encodes ``signal`` with default BSA, decodes it with the same filter and returns the SNR in dB.

    The SNR is measured against the mean-removed signal, so 0 dB is no better than predicting the mean. The first and
    last ``filter_length`` steps are left out, as decoding cannot rebuild them.
    """
    encoder = BSA(as_data(signal))
    spikes = encoder.get_spikes()[0, :, 0]
    filter_values = encoder._fir_filter()["filter_values"]

    normalised = (signal - signal.min()) / np.ptp(signal)
    decoded = lfilter(filter_values, 1.0, spikes.astype(float))

    edge = len(filter_values)
    reference = normalised[edge:-edge]
    error = decoded[edge:-edge] - reference
    return 10 * np.log10(np.sum((reference - reference.mean()) ** 2) / np.sum(error**2))


@pytest.mark.parametrize(
    "name, min_snr_db",
    [("sine", 15), ("two sines", 12), ("chirp", 12), ("step", 15), ("square", 9), ("noisy sine", 9)],
)
def test_bsa_reconstructs_signal(name, min_snr_db):
    assert bsa_snr_db(SIGNALS[name]) >= min_snr_db


def test_bsa_encodes_batch_like_single_series():
    signals = list(SIGNALS.values())
    data = np.stack([as_data(*signals[:3])[0], as_data(*signals[3:])[0]])

    spikes = BSA(data).get_spikes()

    for sample in range(data.shape[0]):
        for feature in range(data.shape[2]):
            single = BSA(as_data(data[sample, :, feature])).get_spikes()[0, :, 0]
            npt.assert_array_equal(spikes[sample, :, feature], single)


def step_forward_snr_db(signal):
    """Encodes ``signal`` with default Step Forward, decodes it and returns the SNR in dB.

    Decoding adds ``threshold`` for each positive spike and subtracts it for each negative one, starting from the first
    value. As for BSA, the SNR is measured against the mean-removed signal.
    """
    data = as_data(signal)
    encoder = StepForward(data)
    spikes = encoder.get_spikes()[0, :, 0]
    threshold = encoder._get_threshold(data)[0, 0]

    decoded = signal[0] + threshold * np.cumsum(spikes)

    reference = signal - signal.mean()
    return 10 * np.log10(np.sum(reference**2) / np.sum((decoded - signal) ** 2))


@pytest.mark.parametrize("name, min_snr_db", [("sine", 20), ("two sines", 11), ("chirp", 8), ("noisy sine", 14)])
def test_step_forward_reconstructs_signal(name, min_snr_db):
    assert step_forward_snr_db(SIGNALS[name]) >= min_snr_db


def test_step_forward_climbs_step_one_threshold_at_a_time():
    spikes = StepForward(as_data(SIGNALS["step"]), threshold=0.25).get_spikes()[0, :, 0]

    expected = np.zeros(T, dtype=np.int8)
    expected[T // 2 : T // 2 + 3] = 1  # Baseline 0 -> 0.25 -> 0.5 -> 0.75, then within a threshold of 1
    npt.assert_array_equal(spikes, expected)


@pytest.mark.parametrize("direction", [1, -1])
def test_tbr_spikes_once_on_step(direction):
    spikes = TBR(as_data(direction * SIGNALS["step"])).get_spikes()[0, :, 0]

    expected = np.zeros(T - 1, dtype=np.int8)
    expected[T // 2 - 1] = direction
    npt.assert_array_equal(spikes, expected)


def test_tbr_spike_sign_follows_slope():
    signal = SIGNALS["sine"]

    spikes = TBR(as_data(signal)).get_spikes()[0, :, 0]

    fired = spikes != 0
    assert np.any(spikes == 1) and np.any(spikes == -1)
    npt.assert_array_equal(spikes[fired], np.sign(np.diff(signal))[fired])


def test_tbr_no_spikes_on_constant_signal():
    spikes = TBR(as_data(np.full(T, 3.0))).get_spikes()

    assert not spikes.any()
