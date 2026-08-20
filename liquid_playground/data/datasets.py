"""Auto-downloading loaders for every dataset used by the model examples/benchmarks.

Every function downloads (once, cached under ``data_cache/``) and returns plain
``torch.Tensor`` train/test splits, so any model in ``models/`` can consume any
dataset here without glue code. See ``datasets/README.md`` for what each dataset
is, why it was picked, and its original source.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _download(url: str, dest: Path, chunk_size: int = 1 << 16) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return dest


# --------------------------------------------------------------------------- #
# 1. Sequential MNIST — image-as-sequence classification, the standard first
#    smoke test for any recurrent/liquid model. Auto-downloads via torchvision.
# --------------------------------------------------------------------------- #
def load_sequential_mnist(permuted: bool = False, seed: int = 0):
    """Returns (train_x, train_y, test_x, test_y) with x shaped (N, 784, 1)."""
    from torchvision import datasets, transforms

    root = CACHE_DIR / "mnist"
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    def to_seq(ds):
        x = ds.data.float().div_(255.0).reshape(len(ds), 784, 1)
        y = ds.targets.long()
        return x, y

    train_x, train_y = to_seq(train_ds)
    test_x, test_y = to_seq(test_ds)

    if permuted:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(784, generator=g)
        train_x = train_x[:, perm, :]
        test_x = test_x[:, perm, :]

    return train_x, train_y, test_x, test_y


# --------------------------------------------------------------------------- #
# 2. UCI Ozone Level Detection — the tabular time-series benchmark used in the
#    original LTC paper (Hasani et al. 2020).
# --------------------------------------------------------------------------- #
_OZONE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data"
)


def load_ozone(test_frac: float = 0.2, seed: int = 0):
    """Returns (train_x, train_y, test_x, test_y). x is (N, 72) features, y is binary."""
    dest = _download(_OZONE_URL, CACHE_DIR / "ozone" / "onehr.data")
    df = pd.read_csv(dest, header=None, na_values="?")
    df = df.dropna()
    x = df.iloc[:, 1:-1].astype("float32").to_numpy()  # drop date col + label col
    y = df.iloc[:, -1].astype("float32").to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    split = int(len(x) * (1 - test_frac))
    train_idx, test_idx = idx[:split], idx[split:]

    return (
        torch.tensor(x[train_idx]),
        torch.tensor(y[train_idx]),
        torch.tensor(x[test_idx]),
        torch.tensor(y[test_idx]),
    )


# --------------------------------------------------------------------------- #
# 3. UCI Room Occupancy Detection — small, real sensor time series, good for
#    a fast binary-classification sequence benchmark.
# --------------------------------------------------------------------------- #
_OCCUPANCY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
)


def load_room_occupancy(seq_len: int = 16):
    """Returns (train_x, train_y, test_x, test_y). x is (N, seq_len, 5) sensor windows."""
    zip_path = _download(_OCCUPANCY_URL, CACHE_DIR / "occupancy" / "occupancy_data.zip")
    out_dir = zip_path.parent
    if not (out_dir / "datatraining.txt").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)

    feat_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

    def windows(csv_name):
        df = pd.read_csv(out_dir / csv_name)
        feats = df[feat_cols].astype("float32").to_numpy()
        labels = df["Occupancy"].astype("float32").to_numpy()
        n = len(feats) // seq_len
        x = feats[: n * seq_len].reshape(n, seq_len, len(feat_cols))
        y = labels[: n * seq_len].reshape(n, seq_len)[:, -1]  # label at window end
        return torch.tensor(x), torch.tensor(y)

    train_x, train_y = windows("datatraining.txt")
    test_x, test_y = windows("datatest.txt")
    return train_x, train_y, test_x, test_y


# --------------------------------------------------------------------------- #
# 4. ETT (Electricity Transformer Temperature) — the standard long-sequence
#    forecasting benchmark used to compare S4/LRU/Mamba-style SSMs, so it's a
#    natural fit for Liquid-S4 / LrcSSM. Hosted as plain CSV on GitHub.
# --------------------------------------------------------------------------- #
_ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def load_ett(seq_len: int = 96, pred_len: int = 24, test_frac: float = 0.2):
    """Returns (train_x, train_y, test_x, test_y) for autoregressive forecasting.

    x: (N, seq_len, 7), y: (N, pred_len, 7) -- 7 = the ETTh1 sensor channels.
    """
    dest = _download(_ETT_URL, CACHE_DIR / "ett" / "ETTh1.csv")
    df = pd.read_csv(dest)
    values = df.drop(columns=["date"]).astype("float32").to_numpy()

    mean, std = values.mean(0, keepdims=True), values.std(0, keepdims=True) + 1e-6
    values = (values - mean) / std

    windows_x, windows_y = [], []
    total = seq_len + pred_len
    for i in range(0, len(values) - total + 1):
        windows_x.append(values[i : i + seq_len])
        windows_y.append(values[i + seq_len : i + total])
    x = np.stack(windows_x)
    y = np.stack(windows_y)

    split = int(len(x) * (1 - test_frac))
    return (
        torch.tensor(x[:split]),
        torch.tensor(y[:split]),
        torch.tensor(x[split:]),
        torch.tensor(y[split:]),
    )


# --------------------------------------------------------------------------- #
# 5. UCI Localization Data for Person Activity — the irregularly-sampled
#    multivariate sequence dataset used directly in the CfC / NCP papers.
# --------------------------------------------------------------------------- #
_PERSON_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt"
)
_PERSON_TAGS = ["010-000-024-033", "010-000-030-096", "020-000-033-111", "020-000-032-221"]
_PERSON_ACTIVITIES = [
    "walking", "falling", "lying down", "lying", "sitting down", "sitting",
    "standing up from lying", "on all fours", "sitting on the ground",
    "standing up from sitting", "standing up from sitting on the ground",
]


def load_person_activity(seq_len: int = 25, test_frac: float = 0.2, seed: int = 0):
    """Returns (train_x, train_y, test_x, test_y). x: (N, seq_len, 7) = xyz + one-hot tag.

    y: (N,) activity class id, label of the last timestep in each window. The
    four body tags are logged asynchronously (no shared timestamps to align
    on), so each raw reading is kept as its own step -- xyz position plus a
    one-hot of which tag it came from -- ordered by timestamp within each
    recording (seq_id), rather than trying to synchronize the four tags.
    """
    dest = _download(_PERSON_URL, CACHE_DIR / "person_activity" / "ConfLongDemo_JSI.txt")
    cols = ["seq_id", "tag_id", "timestamp", "date", "x", "y", "z", "activity"]
    df = pd.read_csv(dest, header=None, names=cols)
    df = df[df["tag_id"].isin(_PERSON_TAGS)]
    df["activity"] = df["activity"].str.lower()
    df = df[df["activity"].isin(_PERSON_ACTIVITIES)]
    act_to_id = {a: i for i, a in enumerate(_PERSON_ACTIVITIES)}
    tag_to_id = {t: i for i, t in enumerate(_PERSON_TAGS)}

    xs, ys = [], []
    for _, seq_df in df.groupby("seq_id"):
        seq_df = seq_df.sort_values("timestamp")
        xyz = seq_df[["x", "y", "z"]].astype("float32").to_numpy()
        tag_onehot = np.eye(len(_PERSON_TAGS), dtype="float32")[seq_df["tag_id"].map(tag_to_id).to_numpy()]
        feats = np.concatenate([xyz, tag_onehot], axis=1)
        labels = seq_df["activity"].map(act_to_id).to_numpy()
        for i in range(0, len(feats) - seq_len + 1, seq_len):
            xs.append(feats[i : i + seq_len])
            ys.append(labels[i + seq_len - 1])

    x = torch.tensor(np.stack(xs))
    y = torch.tensor(np.array(ys), dtype=torch.long)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    split = int(len(x) * (1 - test_frac))
    train_idx, test_idx = idx[:split], idx[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


# --------------------------------------------------------------------------- #
# 6. Speech Commands — raw audio classification, used for the Liquid-S4
#    long-sequence benchmark. Heavier download (~2.3GB), opt-in.
# --------------------------------------------------------------------------- #
def load_speech_commands(subset: str = "training"):
    """Returns a torchaudio.datasets.SPEECHCOMMANDS instance (auto-downloaded)."""
    from torchaudio.datasets import SPEECHCOMMANDS

    root = CACHE_DIR / "speech_commands"
    root.mkdir(parents=True, exist_ok=True)
    return SPEECHCOMMANDS(root=str(root), download=True, subset=subset)
