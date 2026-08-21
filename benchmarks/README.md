# Benchmarks

Runs all eight models -- the five liquid architectures (LTC, CfC, NCP,
Liquid-S4, LrcSSM) plus three non-liquid baselines (RNN, CT-RNN, Neural ODE,
see `docs/model_comparison.md` for why they're included) -- on the *same*
dataset with the same train/eval loop, so accuracy/MSE, parameter count, and
wall-clock train time are directly comparable.

```bash
python benchmarks/run_all.py --config benchmarks/configs/classification_suite.yaml --device auto
python benchmarks/run_all.py --config benchmarks/configs/forecasting_suite.yaml --device auto
python benchmarks/run_all.py --config benchmarks/configs/person_activity_suite.yaml --device auto
```

Each run prints a comparison table and writes a CSV to `benchmarks/results/`
(gitignored -- results are meant to be regenerated on your own hardware).

## JAX vs. PyTorch (CfC only)

CfC is the one model with a JAX/Flax port (`models/cfc/model_jax.py`) as
well as the usual PyTorch one. `jax_vs_pytorch_cfc.py` trains the identical
architecture in both frameworks on the same UCI Person Activity task and
prints wall-clock time per training step and final accuracy side by side:

```bash
python benchmarks/jax_vs_pytorch_cfc.py --device auto --epochs 15
```

Not a rigorous perf study -- no steady-state-only timing, no multiple seeds
-- just a quick, honest "same model, two frameworks, this machine" sanity
check. `first_step_s` is dominated by JAX's one-time JIT/trace compile cost,
which `avg_step_s` will also reflect if `--epochs` is small.

## Suites

| Config | Dataset | Task | Metric |
|---|---|---|---|
| `classification_suite.yaml` | UCI Room Occupancy | binary sequence classification | accuracy |
| `forecasting_suite.yaml` | ETTh1 | long-horizon forecasting | MSE (lower is better) |
| `person_activity_suite.yaml` | UCI Person Activity | 11-way sequence classification | accuracy |

## Add your own combo

Every model factory in `liquid_playground/benchmark/registry.py` has the
signature `(input_size, output_size, **hp) -> nn.Module`, and every dataset
loader returns `(train_x, train_y, test_x, test_y)`, so any model can run on
any dataset in `liquid_playground/data/`. Copy one of the YAML files, swap
`dataset:` and the per-model hyperparameters, and run it.
