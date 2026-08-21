# Getting started

## What `nn.Linear` actually computes

Every model in this repo is hand-written from scratch: the ODE steps, gates,
closed-form updates, sparse wiring, and parallel scans are all literal Python
in `models/*/model.py`, not calls into a pre-built liquid-network library.
The one PyTorch building block every model *does* reuse is `nn.Linear`, so
it's worth being precise about what it is before reading further, since every
equation page below treats it as a primitive:

$$
y = x W^\top + b
$$

`W` has shape `(out_features, in_features)`, `b` has shape `(out_features,)`,
both are `nn.Parameter` tensors initialized `Uniform(-k, k)` with
$k = 1/\sqrt{\text{in\_features}}$. That's the entire computation -- no
activation, nothing hidden. Written out with plain tensors instead of the
`nn.Module` wrapper:

```python
import math
import torch

class MyLinear:
    def __init__(self, in_features, out_features, bias=True):
        k = 1.0 / math.sqrt(in_features)
        self.weight = torch.empty(out_features, in_features).uniform_(-k, k).requires_grad_()
        self.bias = (
            torch.empty(out_features).uniform_(-k, k).requires_grad_() if bias else None
        )

    def __call__(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y
```

`x @ self.weight.T + self.bias` *is* `nn.Linear.forward` -- matmul and
addition are differentiable primitives PyTorch's autograd already knows how
to differentiate, so no backward pass needs to be hand-written either;
`nn.Module`/`nn.Parameter` only add bookkeeping (`.parameters()`, `.to(device)`,
`state_dict()`), not different math. So when a model page below writes
"`Linear w_in`" inside a diagram, it means exactly this one-line operation --
every *other* box in that diagram (the gates, the ODE solves, the scans) is
the part actually implemented by hand for that model.

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
