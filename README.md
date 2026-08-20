# liquid-nn-playground

A playground for **liquid neural networks**: implement, run, and benchmark
five liquid/liquid-adjacent architectures side by side on the same datasets,
with a Python example *and* a notebook example for each.

| Model | Paper | Folder |
|---|---|---|
| **LTC** -- Liquid Time-Constant Networks | Hasani et al., AAAI 2021 | [`models/ltc`](models/ltc) |
| **CfC** -- Closed-form Continuous-time Networks | Hasani et al., Nature MI 2022 | [`models/cfc`](models/cfc) |
| **NCP** -- Neural Circuit Policies (sparse wiring) | Lechner et al., Nature MI 2020 | [`models/ncp`](models/ncp) |
| **Liquid-S4** -- Liquid Structural State-Space Models | Hasani et al., ICLR 2023 | [`models/liquid_s4`](models/liquid_s4) |
| **LrcSSM** -- Liquid-Resistance Liquid-Capacitance SSM (newest, NeurIPS 2025) | Farsang et al., 2025 | [`models/lrcssm`](models/lrcssm) |

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

## Compare all 5 models

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
