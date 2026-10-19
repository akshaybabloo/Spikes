# Test data

EEG from subject S001 of the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (PhysioNet), built by `scripts/make_dataset.py` from runs R04, R08 and R12 (imagine opening and closing the left or right fist).

- 60 samples, `sam1.csv` ... `sam60.csv`, 20 per class
- Each sample is one second after a cue: 160 rows (160 Hz) by 14 columns, in µV
- Channels: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CPz, CP4, Pz
- `tar_class_labels.csv`: 1 = rest (T0), 2 = imagined left fist (T1), 3 = imagined right fist (T2)

To rebuild it:

```
uv run scripts/make_dataset.py
```

## Licence and citation

The dataset is available under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/eegmmidb/view-license/1.0.0/). When using it, please cite:

- Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
