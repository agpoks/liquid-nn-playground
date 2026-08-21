# liquid-nn-playground

A playground for **liquid neural networks** -- implement, run, and benchmark
Liquid Time-Constant Networks (LTC), Closed-form Continuous-time networks
(CfC), Neural Circuit Policies (NCP), Liquid-S4, and the newest
Liquid-Resistance Liquid-Capacitance SSM (LrcSSM), all side by side on the
same datasets.

Every model ships with a runnable Python example and a Jupyter notebook, and
all five share one benchmark harness so accuracy, parameter count, and
training time are directly comparable.

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
model_comparison
models/rnn
models/ctrnn
models/ctgru
models/node
models/liquid_lstm
models/ltc
models/cfc
models/ncp
models/liquid_s4
models/lrcssm
datasets
benchmarks
papers
```
