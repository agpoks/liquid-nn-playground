# Getting started

## Install

```bash
git clone https://github.com/agpoks/liquid-nn-playground.git
cd liquid-nn-playground
pip install -e ".[notebooks]"
```

## Run one model

```bash
python models/ltc/example.py --device auto
```

Every example script accepts `--device {auto,cpu,cuda,mps}` -- `auto` (the
default) picks CUDA or Apple MPS if available and falls back to CPU. Every
model also has a matching `example.ipynb` in the same folder.

## Run a benchmark suite

```bash
python benchmarks/run_all.py --config benchmarks/configs/classification_suite.yaml --device auto
```

This trains all five models on the same dataset and prints/saves a comparison
table (accuracy or MSE, parameter count, wall-clock train time). See
{doc}`benchmarks` for the available suites.

## Datasets

All datasets auto-download on first use -- no manual steps, no accounts. See
{doc}`datasets` for what's available and why each one was picked.

## Pre-fetch everything before going offline

```bash
python datasets/download.py           # datasets
./papers/fetch_papers.sh               # paper PDFs
```
