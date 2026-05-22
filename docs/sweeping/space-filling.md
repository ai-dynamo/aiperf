<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Space-filling sweeps (Sobol, Latin Hypercube)

When grid is too dense and adaptive search is too goal-directed, **quasi-Monte-Carlo (QMC) sweeps** draw a fixed budget of variations that cover the parameter space evenly. Use them for *characterization* — getting a representative scatter you can plot — rather than *optimization*.

## When to use which

| | Sobol | Latin Hypercube |
|---|---|---|
| Marginal axis coverage | very good | perfect (one bin per axis) |
| Joint 2D / 3D coverage | excellent | random-ish |
| Best at | continuous dims, plotting paired comparisons | all-discrete dims, small budgets |
| Sample-count sweet spot | powers of 2 | any |
| Resumable | yes (extend by adding samples) | no |

Default to Sobol unless you have all-discrete dims or want exact per-bin balance.

## Use case 1 — Perf surface across realistic input shapes

```yaml
sweep:
  type: sobol
  samples: 64
  seed: 42
  dimensions:
    - {path: concurrency,                  lo: 1,   hi: 256,   scale: log, kind: int}
    - {path: datasets.default.prompts.isl, lo: 128, hi: 32768, scale: log, kind: int}
    - {path: datasets.default.prompts.osl, lo: 16,  hi: 4096,  scale: log, kind: int}
    - {path: rate,                         lo: 1,   hi: 100,   scale: log, kind: real}
multi_run:
  num_runs: 3
```

64 samples × 3 trials = 192 cells. A 5⁴ grid would be 625 cells — you save ~70% with comparable scatter coverage.

## Use case 2 — A/B build comparison on a fixed workload distribution

Run identical YAML on build A, then build B. The same `seed` makes Sobol pick **identical 32 points** both runs, giving you paired comparisons — much better variance than independent sweeps.

```yaml
sweep:
  type: sobol
  samples: 32
  seed: 42
  scramble: true
  dimensions:
    - {path: concurrency,                  lo: 1,   hi: 128,  scale: log, kind: int}
    - {path: datasets.default.prompts.isl, lo: 256, hi: 8192, scale: log, kind: int}
    - {path: datasets.default.prompts.osl, lo: 64,  hi: 2048, scale: log, kind: int}
```

## Use case 3 — Latin Hypercube on discrete dims

LHS guarantees each `model` appears `samples / len(choices)` times — its niche over Sobol when all dims are discrete.

```yaml
sweep:
  type: latin_hypercube
  samples: 12
  seed: 7
  dimensions:
    - path: model
      choices:
        - meta-llama/Llama-3.1-8B-Instruct
        - mistralai/Mixtral-8x7B-Instruct-v0.1
        - Qwen/Qwen2-72B
    - path: concurrency
      choices: [1, 4, 16, 64]
    - path: datasets.default.prompts.isl
      choices: [256, 1024, 4096]
```

## Reproducibility

Every QMC run writes `<artifacts>/sweep_aggregate/sampling_design.json` containing the mapped sample values (`samples_mapped`), the dimension specs, the sampler type/seed, and sampler options (`scramble` for Sobol, `optimization` for LHS). Re-runs with the same seed and `scipy` version produce byte-identical points. Pin scipy in `pyproject.toml` if you need cross-machine reproducibility over time.

## Edge cases

| | |
|---|---|
| `samples=10` for Sobol | warns; Sobol balances best at powers of 2 (8, 16, 32, 64, ...) |
| `samples=1` | rejected; QMC needs ≥ 2 |
| `lo >= hi` | rejected; pointless or inverted dim |
| `scale: log` and `lo <= 0` | rejected; log requires positive `lo` |
| Both `lo`/`hi` and `choices` set | rejected; use one |

## See also

- [Bayesian-Optimization Outer Loop](bayesian-optimization.md) — when you want optimization, not characterization
- [Bayesian Optimization — 1D SLA saturation](bayesian-optimization.md) — single-dim ramp recipe (`max-concurrency-under-sla` / `max-goodput-under-slo`)
- [Search Recipes](search-recipes.md) — high-level recipes that build sweeps from intent
