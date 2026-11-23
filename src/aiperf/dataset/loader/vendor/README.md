# Vendored third-party sources

Files here are checked in **verbatim** from upstream. Do not edit them.

Their job is to produce byte-identical output to upstream. A hand-adaptation
cannot be mechanically verified, so any fix belongs upstream, followed by a
re-vendor and a SHA bump.

## `speed_bench_prepare.py`

| | |
|---|---|
| Upstream | [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym) `benchmarks/speed-bench/prepare.py` |
| Source URL | `https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/3a9b9eb981ab7a3ea0ebd517cbd4fca1d586768b/benchmarks/speed-bench/prepare.py` |
| Pinned commit | `3a9b9eb981ab7a3ea0ebd517cbd4fca1d586768b` (authored 2026-11-04) |
| Retrieved | 2026-11-04 |
| sha256 | `640de89c585d6e7a7ad8a5e09ef3621c3471e459e992848d1387f43d7e47ec8a` |
| Licence | Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION (header retained verbatim) |

Unmodified. AIPerf adds no changes; the Apache-2.0 §4(b) modification notice
does not apply.

Resolves the placeholder rows in `nvidia/SPEED-Bench` by refetching prompt text
from the 14 upstream source datasets. Consumed by
`aiperf.dataset.loader.speed_bench_public`.

`tests/unit/dataset/loader/test_vendored_speed_bench_prepare.py` holds two
checks. `test_vendored_prepare_script_is_unmodified` pins the sha256 offline, so
an accidental local edit fails CI. `test_vendored_prepare_script_matches_upstream`
is `@pytest.mark.network` and refetches the pinned commit from
`raw.githubusercontent.com`; only that one can establish that the vendored bytes
are upstream's rather than merely unchanged, and it is what backs the
Apache-2.0 §4(b) "Unmodified" claim below. It skips when upstream is
unreachable, so the offline pin remains the contract that always holds.

### Why this file and not `specdec_bench`

The SPEED-Bench dataset card designates `specdec_bench/datasets/speed.py` in
NVIDIA/Model-Optimizer as canonical. This is *not* that file: NeMo-Gym
carries an independent flattened implementation that has already diverged.

The divergence is load-bearing. Canonical writes parquet with a `turns` column;
Gym supports JSONL with `messages=[{"role": ..., "content": ...}]` via
`--format raw`, which is the shape `SpeedBenchRow` validates. Its default
`--format gym` emits Gym-specific rows instead. AIPerf calls the resolver
directly and retains its existing raw-row conversion and atomic cache writes.
The custom loaders accept resolved `messages` JSONL from either the old
NeMo-Skills script or Gym raw output. A `turns`-based output needs conversion
before those loaders can consume it; loading an old file does not regenerate
its prompts with the new resolver.

The resolver retains one behavior worth knowing: the source dispatch in
`_fetch_all_turns_data` ends in `return example` with no terminal `else`, so an
unrecognised source silently yields placeholder text. AIPerf detects this after
the fact by counting unresolved rows rather than trusting the exit status.

The Gym resolver requires v1.1 source references for HLE and StackSelect
throughput prompts. AIPerf pins the matching dataset commit in
`speed_bench_public.py` and stores resolved rows under `speed-bench/v1.1/`.
Source caches are scoped to a resolution call.
