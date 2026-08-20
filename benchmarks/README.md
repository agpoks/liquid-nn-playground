# Benchmarks

Runs all five models (LTC, CfC, NCP, Liquid-S4, LrcSSM) on the *same* dataset
with the same train/eval loop, so accuracy/MSE, parameter count, and
wall-clock train time are directly comparable.

```bash
python benchmarks/run_all.py --config benchmarks/configs/classification_suite.yaml --device auto
python benchmarks/run_all.py --config benchmarks/configs/forecasting_suite.yaml --device auto
python benchmarks/run_all.py --config benchmarks/configs/person_activity_suite.yaml --device auto
```

Each run prints a comparison table and writes a CSV to `benchmarks/results/`
(gitignored -- results are meant to be regenerated on your own hardware).

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
