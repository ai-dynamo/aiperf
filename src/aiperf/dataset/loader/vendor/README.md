# Vendored third-party sources

Files here are checked in **verbatim** from upstream. Do not edit them.

Their job is to produce byte-identical output to upstream. A hand-adaptation
cannot be mechanically verified, so any fix belongs upstream, followed by a
re-vendor and a SHA bump.

## `speed_bench_prepare.py`

| | |
|---|---|
| Upstream | [`NVIDIA-NeMo/Skills`](https://github.com/NVIDIA-NeMo/Skills) `nemo_skills/dataset/speed-bench/prepare.py` |
| Source URL | `https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/5ac8609a56ac941540b10c92e68d556e6343cd4c/nemo_skills/dataset/speed-bench/prepare.py` |
| Pinned commit | `5ac8609a56ac941540b10c92e68d556e6343cd4c` (authored 2026-03-04) |
| Retrieved | 2026-09-01 |
| sha256 | `a551be4df541474e54e21b480022b0cbb66c2da068fda61b2a64bd3223bbbed2` |
| Licence | Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION (header retained verbatim) |

Unmodified. AIPerf adds no changes; the Apache-2.0 §4(b) modification notice
does not apply.

Resolves the placeholder rows in `nvidia/SPEED-Bench` by refetching prompt text
from the 14 upstream source datasets. Consumed by
`aiperf.dataset.loader.speed_bench_public`.

`tests/unit/dataset/loader/test_vendored_speed_bench_prepare.py` pins the sha256
so an accidental local edit fails CI. Verifying against live upstream is a
separate, network-gated check — the pin is the contract that matters offline.

### Why this file and not `specdec_bench`

The SPEED-Bench dataset card designates `specdec_bench/datasets/speed.py` in
NVIDIA/Model-Optimizer as canonical. This is *not* that file: NeMo-Skills
carries an independent flattened implementation that has already diverged.

The divergence is load-bearing. Canonical writes parquet with a `turns` column;
this writes JSONL with `messages=[{"role": ..., "content": ...}]`, which is the
shape `SpeedBenchRow` validates. Switching to canonical would require changing
our row model, so the two are not interchangeable.

Both share one flaw worth knowing: the source dispatch in
`_fetch_all_turns_data` ends in `return example` with no terminal `else`, so an
unrecognised source silently yields placeholder text. AIPerf detects this after
the fact by counting unresolved rows rather than trusting the exit status.
