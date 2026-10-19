# /// script
# requires-python = ">=3.13"
# dependencies = ["kagglehub>=0.3", "mne>=1.10", "numpy>=2.5"]
# ///
"""
Builds the test data in ``tests/test-data/`` from the PhysioNet EEG Motor Movement/Imagery dataset.

Takes subject S001's motor imagery runs (imagine opening and closing the left or right fist), cuts one second after
each cue on 14 channels over the motor cortex, and writes 20 trials per class in the NeuCube layout that
:class:`spikes.utility.ReadCSV` reads: ``sam1.csv`` ... ``sam60.csv`` and ``tar_class_labels.csv``.

Only the three runs used (about 8 MB) are downloaded, from the Kaggle mirror of the dataset, which is much faster than
PhysioNet. Kaggle caches them in ``~/.cache/kagglehub``.

Usage::

    uv run scripts/make_dataset.py
"""

import argparse
from pathlib import Path

import kagglehub
import mne
import numpy as np

KAGGLE_DATASET = "gamalasran/physionet-eeg-motor-movement-imagery"
SUBJECT = "S001"
RUNS = ["R04", "R08", "R12"]  # Imagine opening and closing the left or right fist
CLASSES = {"T0": 1, "T1": 2, "T2": 3}  # Rest, left fist, right fist
TRIALS_PER_CLASS = 20
CHANNELS = ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4", "Pz"]


def load_trials():
    """Returns ``{class_label: [trial, ...]}``, each trial being an array of shape ``(time, channels)`` in µV."""
    trials = {label: [] for label in CLASSES.values()}

    for run in RUNS:
        path = kagglehub.dataset_download(KAGGLE_DATASET, path=f"files/{SUBJECT}/{SUBJECT}{run}.edf")
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        mne.datasets.eegbci.standardize(raw)
        raw.pick(CHANNELS)

        events, _ = mne.events_from_annotations(raw, event_id=CLASSES, verbose="error")
        sfreq = raw.info["sfreq"]
        epochs = mne.Epochs(
            raw, events, event_id=CLASSES, tmin=0, tmax=1 - 1 / sfreq, baseline=None, preload=True, verbose="error"
        )

        for (_, _, label), epoch in zip(epochs.events, epochs.get_data(units="uV")):
            trials[label].append(epoch.T)

    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "tests" / "test-data")
    args = parser.parse_args()

    trials = load_trials()
    for label, items in trials.items():
        if len(items) < TRIALS_PER_CLASS:
            raise SystemExit(f"Class {label} has only {len(items)} trials, need {TRIALS_PER_CLASS}")

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("sam*.csv"):
        old.unlink()

    labels = []
    for label in sorted(trials):
        for trial in trials[label][:TRIALS_PER_CLASS]:
            labels.append(label)
            np.savetxt(args.out / f"sam{len(labels)}.csv", trial, delimiter=",", fmt="%g")
    np.savetxt(args.out / "tar_class_labels.csv", labels, fmt="%d")

    shape = trials[1][0].shape
    print(f"Wrote {len(labels)} samples of {shape[0]} time steps x {shape[1]} channels to {args.out}")


if __name__ == "__main__":
    main()
