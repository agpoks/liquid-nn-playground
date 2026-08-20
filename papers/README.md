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

## Also worth knowing about (not separately modeled here)

- **Neural ODEs** (Chen et al., NeurIPS 2018, [arXiv:1806.07366](https://arxiv.org/abs/1806.07366))
  -- the general "hidden state defined by an ODE" framework LTC specializes.
- **S4** (Gu, Goel, Ré, ICLR 2022, [arXiv:2111.00396](https://arxiv.org/abs/2111.00396))
  -- the structured state-space model Liquid-S4 builds on.
- **ncps** ([github.com/mlech26l/ncps](https://github.com/mlech26l/ncps)) --
  the official, production-grade PyTorch/TensorFlow implementation of LTC, CfC
  and NCP wiring, by the same authors. Use it if you need the exact reference
  numerics; this repo's `models/` implementations are compact educational
  reimplementations aimed at side-by-side comparison and benchmarking.
