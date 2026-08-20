# Datasets

All datasets below are downloaded **automatically** the first time you call the
corresponding loader in `liquid_playground.data` (or run any example script) —
no manual downloads, no account/credential gating. Files are cached in
`data_cache/` at the repo root (gitignored).

| Dataset | Task | Loader | Used well by | Original source |
|---|---|---|---|---|
| Sequential / Permuted MNIST | 10-way image-as-sequence classification | `load_sequential_mnist(permuted=False)` | LTC, CfC, NCP | [torchvision](https://pytorch.org/vision/stable/datasets.html#mnist) (auto) |
| UCI Ozone Level Detection | Binary tabular time-series classification | `load_ozone()` | LTC, CfC | [UCI ML Repo #172](https://archive.ics.uci.edu/dataset/172/ozone+level+detection) — the same benchmark used in the original LTC paper |
| UCI Room Occupancy Detection | Binary sensor sequence classification | `load_room_occupancy(seq_len=16)` | LTC, CfC, NCP | [UCI ML Repo #357](https://archive.ics.uci.edu/dataset/357/occupancy+detection) |
| ETT (ETTh1) | Multivariate long-horizon forecasting | `load_ett(seq_len=96, pred_len=24)` | Liquid-S4, LrcSSM | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) (raw CSV on GitHub) — standard long-sequence SSM benchmark |
| UCI Localization Data for Person Activity | 11-way irregularly-sampled activity classification | `load_person_activity(seq_len=25)` | CfC, NCP | [UCI ML Repo #196](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity) — used directly in the CfC / NCP papers |
| Speech Commands | Raw audio keyword classification (long sequences) | `load_speech_commands()` | Liquid-S4 | [torchaudio](https://pytorch.org/audio/stable/datasets.html#speechcommands) (auto, ~2.3GB, opt-in) |

## Quick use

```python
from liquid_playground.data import load_ozone

train_x, train_y, test_x, test_y = load_ozone()
```

Every `models/<model>/example.py` script picks a sensible default dataset for
that model and exposes `--dataset` to switch between the ones listed above
where the shapes are compatible.

## Why these six

- They cover the three example shapes you actually need: fixed-length
  classification (MNIST), short tabular sequences (Ozone, Occupancy),
  long-horizon forecasting (ETT), irregular multivariate sequences (Person
  Activity), and long raw-signal sequences (Speech Commands).
- Every one of them is a direct HTTP(S) download of a plain file (CSV, ZIP, or
  a standard torchvision/torchaudio dataset) — no manual registration, no
  PhysioNet-style credentialing, no LFS/git-annex.
- Ozone and Person Activity are the same datasets the original LTC/CfC/NCP
  papers benchmark against, so results here are directly comparable to the
  papers in `papers/`.

## Re-downloading

Delete the relevant subfolder under `data_cache/` (or the whole directory) and
re-run any loader — it re-fetches automatically.
