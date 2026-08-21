# Papers

For each model in `models/`, the paper picked here is the one that states its
core equations in the clearest, lowest-complexity form -- usually the original
paper, since in every case below it's also the one with the simplest
self-contained derivation (no follow-up paper states these models more simply
than their own introduction).

PDFs are **not** committed to the repo (keeps it small, avoids redistribution
issues) -- run `./papers/fetch_papers.sh` to download the open-access ones
into `papers/pdfs/` (gitignored). BibTeX for all five is in
[`references.bib`](references.bib).

| Model | Paper | Year | Link | Clearest because |
|---|---|---|---|---|
| [LTC](../models/ltc) | Liquid Time-constant Networks | AAAI 2021 | [arXiv:2006.04439](https://arxiv.org/abs/2006.04439) | Eq. 5-6 state the liquid ODE and its bounded-state proof directly, with no extra machinery |
| [CfC](../models/cfc) | Closed-form Continuous-time Neural Networks | Nature MI 2022 | [arXiv:2106.13898](https://arxiv.org/abs/2106.13898) · [open access](https://www.nature.com/articles/s42256-022-00556-7) | Eq. 8-10 give the exact closed-form gate in one line, replacing LTC's ODE solver |
| [NCP](../models/ncp) | Neural circuit policies enabling auditable autonomy | Nature MI 2020 | [DOI](https://www.nature.com/articles/s42256-020-00237-3) ([TU Wien record](https://repositum.tuwien.at/handle/20.500.12708/141225), no arXiv preprint) | Fig. 1-2 lay out the sensory/inter/command/motor wiring diagram most plainly |
| [Liquid-S4](../models/liquid_s4) | Liquid Structural State-Space Models | ICLR 2023 | [arXiv:2209.12951](https://arxiv.org/abs/2209.12951) | Sec. 3 derives the liquid correction to `A` as a small, explicit addition to the standard S4 recurrence |
| [LrcSSM](../models/lrcssm) | Parallelization of Non-linear State-Space Models: Scaling Up LrcSSM | NeurIPS 2025 | [arXiv:2505.21717](https://arxiv.org/abs/2505.21717) | Sec. 3 states the diagonal-Jacobian RC-circuit update and the resulting parallel-scan solvability in a handful of equations |

## Note on the NCP paper

Unlike the other four, NCP has no arXiv preprint and is paywalled at Nature MI.
The TU Wien institutional record above is metadata-only (no attached PDF at
time of writing) -- use your institutional access (e.g. via TU Wien) or Nature
MI directly.

## Coverage check

The five papers above are the complete core lineage of this specific liquid-network
family (LTC &rarr; CfC &rarr; NCP wiring &rarr; Liquid-S4 &rarr; LrcSSM), all from
the same Hasani/Lechner/Rus/Grosu research line, confirmed against recent
(2025-2026) survey/comparison papers -- there is no missing "6th" architecture
paper in this lineage as of this writing. LrcSSM (May 2025, NeurIPS 2025) is
the newest and currently the last entry in the chain.

Two adjacent things you'll see referenced alongside this lineage but that
aren't separate architectures modeled here:

- **Liquid AI's LFM2 / "Liquid Nanos"** -- the company co-founded by these
  authors' production on-device foundation models (2025). Related in lineage
  and branding, but a different, larger-scale model family with its own
  (largely proprietary) architecture details, not just LTC/CfC at scale.
- Comparative/survey papers such as *"Comparative Analysis of Liquid Neural
  Networks and LSTM..."* ([arXiv:2605.27467](https://arxiv.org/abs/2605.27467))
  benchmark the architectures above rather than introducing new ones.

## Baseline architectures (for comparison, not liquid networks)

Three non-liquid baselines live alongside the five liquid models specifically
to show what each additional piece of machinery buys, one step at a time:
plain discrete-time recurrence -> fixed continuous time -> fully general
learned continuous-time dynamics -> LTC's *input-gated, structured* ODE
(where "liquid" starts). Same file layout as the liquid models
(`model.py` / `example.py` / `example.ipynb` / `README.md`), same UCI Ozone
task in every `example.py` as [`models/ltc`](../models/ltc) so results are
directly comparable.

| Model | Paper | Year | Link | What it adds / lacks vs. LTC |
|---|---|---|---|---|
| [RNN](../models/rnn) | Finding Structure in Time | Cognitive Science 1990 | classic reference, no arXiv/DOI-linkable preprint | No time-awareness at all -- discrete step, no `dt`, no time constant |
| [CT-RNN](../models/ctrnn) | Approximation of dynamical systems by continuous time recurrent neural networks | Neural Networks 1993 | [DOI](https://doi.org/10.1016/S0893-6080(05)80125-X) | Continuous-time ODE, but the leak's time constant is fixed, never input-dependent |
| [Neural ODE](../models/node) | Neural Ordinary Differential Equations | NeurIPS 2018 | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) | Fully general learned `dh/dt = f_theta(h, x)`, no structure or stability guarantee -- LTC is a specific, bounded case of this |

## Liquid NNs in the wild -- further reading

Papers above are the primary sources; these are more accessible write-ups and
real deployments, useful for building intuition or seeing the models used
outside a benchmark table:

- [Drones navigate unseen environments with liquid neural networks](https://www.csail.mit.edu/news/drones-navigate-unseen-environments-liquid-neural-networks)
  (MIT CSAIL, 2023) -- the flagship real-world result: CfC-piloted drones
  generalizing to forests and urban scenes never seen in training.
- [Applying Liquid Neural Networks (LNN) in Self-Driving Labs (SDL)](https://medium.com/@isissifeng/applying-liquid-neural-networks-lnn-in-self-driving-labs-sdl-837447b7df5e)
  (Sissi Feng, Medium) -- LNNs for closed-loop, adaptive control in autonomous
  chemistry/materials labs, a domain outside vision/robotics.
- [Neural Circuit Policy: training autonomous vehicles using models inspired by the nervous system](https://ved933409.medium.com/neural-circuit-policy-training-a-autonomous-vehicles-using-models-inspired-by-nervous-system-db79a554ebef)
  (Ved Prakash, Medium) -- compares random, fully-connected, and NCP wiring
  head-to-head on a driving task; a good companion to [`models/ncp`](../models/ncp).
- [TinyML -- Liquid Neural Networks](https://medium.com/@thommaskevin/tinyml-liquid-neural-networks-e5978f222dd7)
  (thommaskevin, Medium) -- walks the LTC ODE and fixed-step Euler update by
  hand with worked numbers, good if the equations in [`models/ltc`](../models/ltc) move too fast.
- [Liquid Neural Nets (LNNs): a deep dive](https://medium.com/@hession520/liquid-neural-nets-lnns-32ce1bfb045a)
  (Jake Hession, Medium) -- broader intuition-first tour of why time-continuous,
  input-dependent dynamics behave differently from LSTM/GRU.
- [Liquid Neural Networks](https://abdulkaderhelwan.medium.com/liquid-neural-networks-37ccaaee469a)
  (Abdulkader Helwan, Medium) -- short, accessible overview of the "equations
  that adapt to new inputs" framing, a good first read before the papers above.

## Also worth knowing about (not separately modeled here)

- **S4** (Gu, Goel, Ré, ICLR 2022, [arXiv:2111.00396](https://arxiv.org/abs/2111.00396))
  -- the structured state-space model Liquid-S4 builds on.
- **ncps** ([github.com/mlech26l/ncps](https://github.com/mlech26l/ncps)) --
  the official, production-grade PyTorch/TensorFlow implementation of LTC, CfC
  and NCP wiring, by the same authors. Use it if you need the exact reference
  numerics; this repo's `models/` implementations are compact educational
  reimplementations aimed at side-by-side comparison and benchmarking.
