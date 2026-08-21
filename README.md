<p align="center">
  <img src="docs/source/_static/logo-banner.svg" alt="liquid-nn-playground" width="520">
</p>

# liquid-nn-playground

A playground for **liquid neural networks**: implement, run, and benchmark
five liquid/liquid-adjacent architectures side by side on the same datasets,
with a Python example *and* a notebook example for each, plus four
non-liquid baselines and one hybrid that isolate exactly what "liquid" adds.

| Model | Paper | Folder |
|---|---|---|
| **LTC** -- Liquid Time-Constant Networks | Hasani et al., AAAI 2021 | [`models/ltc`](models/ltc) |
| **CfC** -- Closed-form Continuous-time Networks | Hasani et al., Nature MI 2022 | [`models/cfc`](models/cfc) |
| **NCP** -- Neural Circuit Policies (sparse wiring) | Lechner et al., Nature MI 2020 | [`models/ncp`](models/ncp) |
| **Liquid-S4** -- Liquid Structural State-Space Models | Hasani et al., ICLR 2023 | [`models/liquid_s4`](models/liquid_s4) |
| **LrcSSM** -- Liquid-Resistance Liquid-Capacitance SSM (newest, NeurIPS 2025) | Farsang et al., 2025 | [`models/lrcssm`](models/lrcssm) |

**Baselines** (not liquid, included to show what each piece of machinery buys -- see [`docs/`](docs) "How each model treats time"):

| Model | Paper | Folder |
|---|---|---|
| **RNN** -- vanilla Elman RNN, no time-awareness at all | Elman, Cognitive Science 1990 | [`models/rnn`](models/rnn) |
| **CT-RNN** -- continuous time, but a fixed (non-liquid) time constant | Funahashi & Nakamura, Neural Networks 1993 | [`models/ctrnn`](models/ctrnn) |
| **CT-GRU** -- a bank of fixed time constants with learned soft selection | Mozer et al., 2017 | [`models/ctgru`](models/ctgru) |
| **Neural ODE** -- continuous time, fully general learned dynamics | Chen et al., NeurIPS 2018 | [`models/node`](models/node) |

**Hybrid** (liquid, but not from a paper -- built for this repo after confirming no such architecture exists in the literature):

| Model | Built from | Folder |
|---|---|---|
| **Liquid-LSTM** -- LSTM's 4 gates + LTC's continuous-time leak | Hochreiter & Schmidhuber 1997 + Hasani et al. 2021 | [`models/liquid_lstm`](models/liquid_lstm) |

Full paper references and why each one was picked: [`papers/README.md`](papers/README.md).
Docs: see [`docs/`](docs) (built on Read the Docs).

## Layout

```
liquid-nn-playground/
├── models/<name>/       model.py, example.py, example.ipynb, README.md  (one per architecture)
├── liquid_playground/    shared package: device (cpu/gpu/mps) resolution, dataset loaders, benchmark runner
├── datasets/             dataset docs + a one-shot pre-download script
├── benchmarks/           YAML suites that run all 5 models on the same dataset and compare
├── papers/               reference list, BibTeX, PDF-fetch script
└── docs/                 Sphinx / Read the Docs source
```

## Install

```bash
git clone https://github.com/agpoks/liquid-nn-playground.git
cd liquid-nn-playground
pip install -e ".[notebooks]"
```

## Run a model

```bash
python models/ltc/example.py --device auto
```

Every example script (and the benchmark runner) takes `--device {auto,cpu,cuda,mps}`.
`auto` (the default) picks CUDA or Apple Silicon MPS if available, otherwise
CPU -- so the same command works unchanged on a laptop or a GPU box. Every
model also has a matching `example.ipynb` you can open in Jupyter.

CfC also ships a **JAX/Flax** port (`models/cfc/model_jax.py` +
`example_jax.py`, `pip install -e ".[jax]"`) with JAX's own
`--device {auto,cpu,gpu,tpu}` selection, alongside the PyTorch version -- a
quick way to compare the two frameworks on the identical architecture without
re-implementing all five models twice. `benchmarks/jax_vs_pytorch_cfc.py`
trains both back to back and prints step time + accuracy side by side:

```bash
python benchmarks/jax_vs_pytorch_cfc.py --device auto --epochs 15
```

## Compare all 10 models

```bash
python benchmarks/run_all.py --config benchmarks/configs/classification_suite.yaml --device auto
python benchmarks/run_all.py --config benchmarks/configs/forecasting_suite.yaml --device auto
python benchmarks/run_all.py --config benchmarks/configs/person_activity_suite.yaml --device auto
```

Prints and saves (`benchmarks/results/*.csv`) a table of accuracy/MSE,
parameter count, and wall-clock train time per model.

## Datasets

Everything auto-downloads on first use (no accounts, no manual steps) --
see [`datasets/README.md`](datasets/README.md) for the full list (Sequential
MNIST, UCI Ozone, UCI Room Occupancy, ETTh1, UCI Person Activity, Speech
Commands) and why each was picked. Pre-fetch them all with:

```bash
python datasets/download.py
```

## Scope note

The five `models/*/model.py` implementations are compact, readable
reimplementations of each paper's core mechanism, built for **side-by-side
comparison and benchmarking**, not for reproducing every paper's exact
numbers. Every model README links to the authors' official repo for that.

## License

MIT, see [`LICENSE`](LICENSE).
