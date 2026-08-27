<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI flag surface parity audit

**Python baseline: `origin/main` @ `bc359bf8fd`, read at
`/mnt/4tb/aiperf-parity-py-main/src/aiperf/`.** All Python `path:line`
citations below are rooted at that worktree and are relative to `src/aiperf/`.
Rust citations are rooted at the branch checkout
`/home/anthony/nvidia/projects/aiperf/ajc/rust/` and are relative to `rust/`.

This report was originally written against the working-tree Python on branch
`ajc/native-rust-runtime-plugins`, which carries 132 locally-modified Python
files and is not what users ship. It has been fully re-verified against the
baseline above; see `## Withdrawn after baseline correction` for what the
correction changed.

Domain: the public CLI flag surface of `aiperf profile` and the other shared
commands.

## Summary

Flag *name* parity against the real baseline is near-total but not total: of the
301 long names and aliases `origin/main` publishes, Rust accepts 299. The two it
does not accept are `--transport` and `--transport-type`, which is a loud clap
failure rather than a silent one but is a genuine user-visible drop (Finding 12).
The substantive risk is resolution, not naming: Rust re-derives most defaults
inside `rust/cli/src/load.rs` instead of deferring to the Config-v2 schema, and
that re-derivation diverges from baseline in eleven verified places. The
headline item is new and severe — a bare `--user-centric-rate` + `--num-users`
run is bounded at 10 requests by Python and is completely unbounded in Rust, so
the benchmark simply never terminates. Below that sit two lost input
normalizations (Python's threshold-based Hz/kHz sample-rate converter, and its
auto-promotion of `--arrival-smoothness`/`--vllm-burstiness` to a gamma arrival
process), a genuinely unwired `--auto-plot`, a family of "companion flag
silently dropped when its partner is absent" bugs, and four plain default flips
(`--trajectory-start-{min,max}-ratio` 0.25/0.75→0.0/0.0,
`--fixed-schedule-auto-offset` False→True, `--video-duration` 5.0s→1.0s, and the
`--rankings-*`/`--audio-length-mean` companion defaults).

The baseline correction changed the specifics of two findings and withdrew none
of them; it also resolved four of the six items previously listed as unverified,
two of them into new P1 findings. None of the fourteen findings appear in
`docs/dev/python-rust-parity-gaps.md`. That document's P1.4 ("native profile
accepts dead or differently defined flags") is KNOWN(partially-fixed): its named
examples `--workers-max` and the `--zmq-*` set now emit warnings through Rust's
`UNIMPLEMENTED_FLAGS` table.

## Findings

### 1. A bare user-centric run is 10-request-bounded in Python and unbounded in Rust

**Severity:** P0
**Status:** NEW (resolves a previously-unverified item)

**Python evidence** — `config/flags/_converter_profiling.py:552-560`. A
`USER_CENTRIC` phase carries `users`/`rate` in `prof`, never
`requests`/`duration`/`sessions`, so the fallback fires:

```python
    if (
        not any(k in prof for k in ("requests", "duration", "sessions"))
        and prof["type"] != PhaseType.FIXED_SCHEDULE
    ):
        # Why: when no bound is given for an unbounded run, default to
        # 10 requests so the run terminates in a reasonable time.
        # Deliberate override of the PhaseConfig default (which would
        # leave it unbounded).
        prof.setdefault("requests", 10)
```

The predicate excludes only `FIXED_SCHEDULE`. User-centric is bounded at 10
requests.

**Rust evidence** — `rust/runtime/src/config/resolve.rs:1353-1359` excludes
user-centric from the very same fallback:

```rust
    let effective_requests = inputs.request_count.or_else(|| {
        (inputs.benchmark_duration.is_none()
            && inputs.fixed_schedule.is_none()
            && inputs.user_centric.is_none()
            && inputs.sessions.is_none())
        .then_some(DEFAULT_REQUEST_COUNT)
    });
```

and the user-centric phase is then built from the *raw* count, not
`effective_requests` — `rust/runtime/src/config/resolve.rs:1365-1375`:

```rust
    let profiling = if let Some((rate, users)) = inputs.user_centric {
        Phase {
            common: PhaseCommon {
                ...
                requests: inputs.request_count,
                sessions: inputs.sessions,
                duration: inputs.benchmark_duration,
```

With no `--request-count`, `--benchmark-duration`, or session bound, all three
are `None`. No later stage supplies one: `rust/runtime/src/config/phase_validate.rs:235-249`
overlays a bound only when one was authored.

**Observable user impact:**
`aiperf profile --user-centric-rate 5 --num-users 8 -m <model> -u <url>`
completes after 10 requests under Python and never terminates under Rust — it
runs until the user interrupts it, with no error, warning, or progress bound.
There is no way to discover this from the CLI; the flag combination is the
documented way to select user-centric mode. Rated P0 because the tool is
unusable for that invocation and the failure mode (an apparent hang) does not
point at its cause.

**Confidence:** High. Both predicates read directly; the absence of a
compensating bound checked in `phase_validate.rs`.

### 2. Python's threshold-based Hz/kHz sample-rate normalizer is absent in Rust, so kHz-shaped values land 1000× low

**Severity:** P1
**Status:** CHANGED (the divergence is real but narrower and differently shaped
than originally reported — see Withdrawn section)

**Python evidence** — `config/flags/_converter_dataset.py:28-38` normalizes
*conditionally*, accepting both unit spellings:

```python
def _normalize_sample_rate_khz(value: float | int) -> float:
    """Auto-convert Hz inputs to kHz for the kHz-scoped audio schema.

    Pre-redesign cyclopts CLI flags accepted Hz-shaped values like ``16000``
    while the kHz schema caps at 96 (96 kHz = pro audio). Auto-divide
    values above the cap by 1000 to preserve the historical invocation
    shape. Why: chaos suite + tutorials still pass ``16000`` for 16 kHz
    speech audio.
    """
    v = float(value)
    return v / 1000.0 if v > 96.0 else v
```

Applied to `--audio-sample-rates` at `config/flags/_converter_dataset.py:165-168`
and to `--video-audio-sample-rate` at `:220-221`. The flag is documented in kHz
(`config/flags/cli_config.py:1592`, "A list of audio sample rates to randomly
select from in kHz. Common sample rates are 16, 44.1, 48, 96") and the generator
scales the stored kHz value back up (`dataset/generator/audio.py:157`):

```python
            self._format_rng.numpy_choice(self.config.sample_rates) * 1000
```

So Python maps both `16` and `16000` to 16 kHz, and both `44.1` and `44100` to
44.1 kHz.

**Rust evidence** — `rust/cli/src/load.rs:1149-1158` divides
*unconditionally*, having assumed a Hz-only input contract (`rust/cli/src/flags.rs:1132`
documents the flag in Hz):

```rust
    // The wire carries sample rates in kHz (Hz / 1000).
    let sample_rates = if flags.audio_sample_rates.is_empty() {
        vec![16.0]
    } else {
        flags
            .audio_sample_rates
            .iter()
            .map(|r| r / 1000.0)
            .collect()
    };
```

`--video-audio-sample-rate` has the same shape at `rust/cli/src/load.rs:1197-1200`
(`.map(|r| r / 1000.0).unwrap_or(44.1)`).

**Observable user impact:** The two implementations *agree* for Hz-shaped values
above 96 (`--audio-sample-rates 16000` → 16 kHz on both). They diverge by 1000×
for every value at or below 96 — which is exactly the set Python's own help text
names as the common choices. `--audio-sample-rates 16` produces 16 kHz speech
audio under Python and 16 Hz under Rust; `--audio-sample-rates 44.1` produces
44.1 kHz versus 44.1 Hz. Audio payload sizes differ by three orders of
magnitude, moving every byte-count, request-latency, and throughput number on an
audio benchmark. The omitted case agrees (both default to 16 kHz), so only users
who set the flag are affected. `--video-audio-sample-rate 48` diverges
identically.

**Confidence:** High. Normalizer, both call sites, the kHz-scoped schema
contract, and the downstream ×1000 all read on the baseline; the Rust
arithmetic read directly.

### 3. `--arrival-smoothness` / `--vllm-burstiness` no longer auto-promotes the arrival process to gamma; Rust stays Poisson and drops the value

**Severity:** P1
**Status:** NEW

**Python evidence** — `config/flags/_converter_profiling.py:228-238`. The
comment records that non-promotion was already found and fixed once as a
regression:

```python
        # v1 parity (user_config.py auto-promote): --arrival-smoothness /
        # --vllm-burstiness without an explicit --arrival-pattern resolves to
        # gamma, since smoothness is a gamma-distribution knob. Without this the
        # flag fell through to POISSON and then _apply_phase_specific_routes
        # hard-rejected it ("only supported with gamma") -- a cutover regression
        # that made --vllm-burstiness unusable on its own. A 'smoothness'
        # search-space dimension is the same knob, so it auto-promotes too.
        if "arrival_pattern" not in cli.model_fields_set and (
            cli.arrival_smoothness is not None or "smoothness" in search_dims
        ):
            return PhaseType.GAMMA
```

**Rust evidence** — `rust/runtime/src/config/resolve.rs:2072-2081` selects the
phase kind from `rate_mode` alone, and `smoothness` is a field only of the
`Gamma` variant:

```rust
    let kind = if let Some(rate) = rate {
        match rate_mode {
            Some("gamma") => PhaseKind::Gamma {
                rate,
                concurrency,
                smoothness,
            },
            Some("constant") => PhaseKind::Constant { rate, concurrency },
            // Poisson is the default arrival distribution.
            _ => PhaseKind::Poisson { rate, concurrency },
        }
    } else {
```

`rate_mode` comes only from the two explicit mode flags —
`rust/cli/src/load.rs:536-540`
(`rate_mode: flags.request_rate_mode.clone().or_else(|| flags.arrival_pattern.clone())`)
— so nothing consults `smoothness` when choosing the variant. With no explicit
mode flag the run takes the `_ => PhaseKind::Poisson` arm and the authored
smoothness is discarded.

**Observable user impact:**
`aiperf profile --request-rate 10 --arrival-smoothness 2 ...` drives a
gamma-distributed arrival process under Python and a Poisson one under Rust,
with the smoothness value silently dropped. Inter-arrival burstiness is the
entire point of the flag, so queueing delay, TTFT tail percentiles, and p99
latency all differ. This is most likely to bite users porting vLLM benchmark
command lines, since `--vllm-burstiness` is the compatibility alias and is
useless without the promotion — the exact failure Python's comment says was
already regressed once and fixed.

**Confidence:** High.

### 4. `--auto-plot` is a true no-op in Rust while Python runs the plotter

**Severity:** P1
**Status:** STILL VALID

**Python evidence** — declared at `config/flags/cli_config.py:2739` (default
`None`), stored at `config/artifacts.py:178`, and acted on at
`cli_runner/__init__.py:67-71`:

```python
    if plan.configs[0].artifacts.auto_plot:
        from aiperf.plot.auto_plot import build_auto_plot_callback
        ...
            build_auto_plot_callback(
```

Baseline additionally makes `--plot` imply it — `config/config.py:1061`
(`_plot_implies_auto_plot`).

**Rust evidence** — `rust/cli/src/flags.rs:559-566` is the only occurrence of the
field in the entire Rust tree; a repo-wide `rg auto_plot rust/ -g '*.rs'` returns
this definition and one unrelated comment in `rust/cli/src/yaml.rs`:

```rust
    /// Auto-generate plots after the run (`--auto-plot` / `--no-auto-plot`).
    #[arg(
        long = "auto-plot", num_args = 0..=1, default_missing_value = "true",
        overrides_with = "no_auto_plot")]
    pub auto_plot: Option<bool>,
```

`--auto-plot` is absent from `UNIMPLEMENTED_FLAGS`
(`rust/cli/src/profile.rs:383-417`), whose own doc states the invariant it exists
to uphold, while its dependent modifier `--plot-required` *is* listed at
`rust/cli/src/profile.rs:398`:

```rust
/// Entries leave this table by gaining a consumer, not by being deleted: dropping
/// one returns the flag to silently-ignored, which is the failure this guards.
...
    ("--plot-required", |f| f.plot_required.is_some()),
```

**Observable user impact:** `aiperf profile --auto-plot ...` exits 0 with no plot
files and no warning. Pairing it with `--plot-required` warns about
`--plot-required` only, which actively misleads: it implies plotting is live and
merely its failure-escalation is not. Any "benchmark then read the PNGs" script
gets an empty directory.

**Confidence:** High. Proven by absence across the Rust tree, corroborated by the
modifier's presence in the warn table.

### 5. `--trajectory-start-min-ratio` / `--trajectory-start-max-ratio` default 0.25/0.75 in Python and 0.0/0.0 in Rust

**Severity:** P1
**Status:** NEW (resolves a previously-unverified item; the branch tree had these
as `default=None`, which is why this could not be settled before)

**Python evidence** — `config/flags/cli_config.py:2298-2312` and `:2314-2329`.
Both are plain `float` with real defaults:

```python
    trajectory_start_min_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="AGENTIC_REPLAY only: lower bound (inclusive) on the random start "
            "position within each trajectory, expressed as a fraction of the "
            "trace's total turn count. Sampled per trajectory at trajectory-build "
            "time; deterministic given --random-seed.",
    ] = 0.25
```

and `] = 0.75` for the max at `:2329`, whose help adds "The effective per-trace
ceiling is `min(int(max_ratio * n), n - 2)` so at least one profile turn remains
after warmup."

**Rust evidence** — `rust/cli/src/load.rs:585-590` collapses both to 0.0:

```rust
                                trajectory_start_min_ratio: flags
                                    .trajectory_start_min_ratio
                                    .unwrap_or(0.0),
                                trajectory_start_max_ratio: flags
                                    .trajectory_start_max_ratio
                                    .unwrap_or(0.0),
```

Consumed as non-optional `f64` at `rust/runtime/src/config/resolve.rs:471-473`
and `:1688-1689`. A `--scenario` can overwrite them
(`rust/runtime/src/config/resolve.rs:1944-1949`, defaulting max to 1.0), but a
scenario-free agentic-replay run keeps 0.0/0.0.

**Observable user impact:** On an agentic-replay run without `--scenario`, Python
samples each trajectory's start position uniformly from the middle half of its
turns (`[0.25n, 0.75n]`) while Rust starts every trajectory at turn 0. Because
accumulated conversation context grows with turn index, the two benchmark
materially different input-length distributions: Rust measures only cold,
short-context early turns. ISL, prefill cost, TTFT, and cache-hit rate all shift,
with no notice that the requested sampling window was ignored.

**Confidence:** High.

### 6. `--user-centric-rate` alone selects user-centric mode in Python and is silently discarded in Rust

**Severity:** P1
**Status:** STILL VALID

**Python evidence** — `config/flags/_converter_profiling.py:202` plus the
comment at `:205-215`, which states the intent explicitly:

```python
    user_centric_selected = cli.user_centric_rate is not None or user_centric_needed
    ...
    # Gated on user_centric_selected (not
    # just user_centric_needed) so an explicit --user-centric-rate without
    # a 'users' dimension is caught too: it still resolves to USER_CENTRIC,
```

and `:221-222` returns `PhaseType.USER_CENTRIC`. The mirror direction is a loud
refusal at `:431-435`:

```python
    if "num_users" in fields_set and phase_type != PhaseType.USER_CENTRIC:
        raise ValueError(
            "--num-users requires --user-centric-rate. Pass --user-centric-rate "
```

**Rust evidence** — `rust/cli/src/load.rs:224-227` requires both and drops the
pair otherwise:

```rust
    let user_centric_cli = match (flags.user_centric_rate, flags.num_users) {
        (Some(rate), Some(users)) => Some((rate, users)),
        _ => None,
    };
```

**Observable user impact:** `aiperf profile --user-centric-rate 5 ...` runs a
user-centric workload in Python; in Rust it silently degrades to the default
bounded concurrency run (10 requests, no user-centric arrival process) and
reports numbers describing a workload the user did not request. The mirror case
`--num-users 8` alone is a clear Python error and a silent Rust drop — the same
defect in the less damaging direction.

**Confidence:** High.

### 7. `--isl-stddev` is silently discarded when `--isl` is omitted

**Severity:** P1
**Status:** STILL VALID (citations corrected: the baseline field is
`prompt_input_tokens_mean`, not the branch's `synthetic_input_tokens_mean`)

**Python evidence** — the mean and stddev are independent fields with
independent defaults, `config/flags/cli_config.py:1340-1359` (`] = 550`, aliases
`--prompt-input-tokens-mean` / `--synthetic-input-tokens-mean` / `--isl`) and
`:1361-1380` (`] = 0.0`, aliases `--prompt-input-tokens-stddev` /
`--synthetic-input-tokens-stddev` / `--isl-stddev`). The converter forwards each
independently — `config/flags/_converter_dataset.py:55-67`:

```python
    isl: dict[str, Any] = {}
    if "prompt_input_tokens_mean" in s:
        ...
        isl["mean"] = v[0] if isinstance(v, list) and v else v
    if "prompt_input_tokens_stddev" in s:
        v = cli.prompt_input_tokens_stddev
        isl["stddev"] = v[0] if isinstance(v, list) and v else v
    if isl:
        prompts["isl"] = isl
```

So `--isl-stddev 128` alone emits `isl = {"stddev": 128.0}` and the mean comes
from the schema default.

**Rust evidence** — `rust/cli/src/load.rs:473-485` reads the stddev only inside
the `Some(mean)` arm:

```rust
                    isl: match isl_mean {
                        Some(mean) => Distribution {
                            mean: Some(mean),
                            stddev: Some(flags.isl_stddev.unwrap_or(0.0)),
                            ..Default::default()
                        },
                        None => default_isl(),
                    },
                    osl: osl_mean.map(|mean| Distribution {
                        mean: Some(mean),
                        stddev: Some(flags.osl_stddev.unwrap_or(0.0)),
                        ..Default::default()
                    }),
```

`default_isl()` pins mean 550 / stddev 0.0, so the authored stddev never reaches
the config.

**Observable user impact:** A user varying input-length variance without moving
the mean gets zero variance and no warning — every prompt is exactly 550 tokens,
reported ISL standard deviation is 0, and any conclusion about
variance-sensitivity is invalid. The `--osl-stddev`-without-`--osl` case is
structurally identical in Rust (`osl` is `None` entirely), but Python's own help
text at `config/flags/cli_config.py:1499` says that combination "Only applies
when `--prompt-output-tokens-mean` is set" even though the converter forwards it
at `:72-79`, so I am not asserting the OSL half as a divergence — see Unverified.

**Confidence:** High for the ISL half on both sides.

### 8. `--fixed-schedule-auto-offset` default flips from `False` to `True`

**Severity:** P1
**Status:** STILL VALID

**Python evidence** — `config/flags/cli_config.py:832-843`:

```python
    fixed_schedule_auto_offset: Annotated[
        bool,
        Field(
            description="Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. "
            ...
    ] = False
```

**Rust evidence** — `rust/cli/src/load.rs:307-313` derives `true` whenever
neither explicit offset is present:

```rust
        let default_auto = flags.fixed_schedule_start_offset.is_none()
            && flags.fixed_schedule_end_offset.is_none();
        (
            Some(flags.fixed_schedule_auto_offset.unwrap_or(default_auto)),
            Some(count),
        )
```

**Observable user impact:** `--fixed-schedule --input-file trace.jsonl` with no
offsets replays the trace's absolute timestamps under Python and a
zero-normalized schedule under Rust. For a captured production trace whose first
timestamp is not 0 — the common case — Python waits out the leading offset before
issuing anything while Rust starts immediately. Wall-clock run duration, the time
axis of every timeslice export, and any comparison against a previously-recorded
Python run all shift.

**Confidence:** High.

### 9. `--video-duration` default drops from 5.0s to 1.0s

**Severity:** P1
**Status:** STILL VALID

**Python evidence** — `config/flags/cli_config.py:1759-1771`:

```python
    video_duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count "
            ...
    ] = 5.0
```

**Rust evidence** — `rust/cli/src/load.rs:1214`:

```rust
        duration: flags.video_duration.unwrap_or(1.0),
```

**Observable user impact:** A video run that sets some other `--video-*` flag
(e.g. `--video-width`/`--video-height`) but leaves duration alone generates 4
frames instead of 20 at the shared default `--video-fps 4` — a 5× smaller
payload, changing uploaded bytes, request latency, and throughput with no notice.
Both sides agree on `--video-fps 4`, `--video-format webm`,
`--video-codec libvpx-vp9`, `--video-synth-type moving_shapes`,
`--video-audio-depth 16`, and `--video-audio-num-channels 0`
(`config/flags/cli_config.py:1785`, `:1889`, and the Rust defaults at
`rust/cli/src/load.rs:1192-1226`), which makes the lone duration mismatch look
like an oversight rather than a redesign.

**Confidence:** High.

### 10. Naming one flag in a media/rankings group silently re-specifies its companions' defaults

**Severity:** P1
**Status:** STILL VALID

**Python evidence** — every field carries its own independent default, so setting
one does not disturb the others. `config/flags/cli_config.py:1938`
(`rankings_passages_mean = 1`), `:1965`
(`rankings_passages_prompt_token_mean = 550`), `:1992`
(`rankings_query_prompt_token_mean = 550`), and `:1545`
(`audio_length_mean = 0.0`).

**Rust evidence** — `rust/cli/src/load.rs:1068-1112`. `build_rankings` returns
`None` unless at least one rankings flag is set; once *any* is set, every
sub-distribution the user did not name falls back to a hardcoded value that does
not match Python:

```rust
    Some(crate::model::dataset::Rankings {
        passages: rankings_dist(..., 10.0),
        passage_tokens: rankings_dist(..., 128.0),
        query_tokens: rankings_dist(..., 32.0),
    })
```

The audio spec has the same shape — `rust/cli/src/load.rs:1141-1148` falls back
to `default_media_dim()`, which is a fixed 512 (`rust/cli/src/load.rs:855-860`):

```rust
    let length = match flags.audio_length_mean {
        Some(mean) => Distribution { ... },
        None => default_media_dim(),
    };
```

**Observable user impact:** `--rankings-passages-mean 5` against a `rankings`
endpoint gives 550-token passages and a 550-token query in Python, and 128-token
passages with a 32-token query in Rust — roughly 4× and 17× smaller, so ISL,
prefill cost, and throughput are reported against a much smaller workload.
Independently, `--audio-batch-size 2` with no explicit length gives 0.0-length
audio in Python and 512-unit-length audio in Rust. The coupling is the
surprising part: naming one flag in the group silently re-specifies the others.

**Confidence:** High on both sides' resolution code. Medium on the exact
end-to-end token/byte counts, which depend on downstream composition not
runtime-verified.

### 11. Python's per-loader preferred-sampling override has no Rust counterpart

**Severity:** P1
**Status:** CHANGED (mechanism confirmed identically on baseline; the blast
radius is narrower than originally stated — it is two opt-in dataset types, not
all synthetic runs)

**Python evidence** — `--dataset-sampling-strategy` defaults to `None`
(`config/flags/cli_config.py:725-738`) and is forwarded only when explicitly set
(`config/flags/_converter_dataset.py:260`). The schema default is `SEQUENTIAL`
(`config/dataset/config.py:129-132`), which is then overridden by the loader's
preference at `config/dataset/resolver.py:265-277`:

```python
    @staticmethod
    def _resolve_sampling(ds: object, dataset_type: object) -> object:
        """Pick the loader's preferred sampling unless the user set an explicit one."""
        loader_sampling = DatasetResolver._get_preferred_sampling(dataset_type)
        ds_sampling = ds.sampling  # type: ignore[attr-defined]
        if (
            ds_sampling == DatasetSamplingStrategy.SEQUENTIAL
            and loader_sampling != DatasetSamplingStrategy.SEQUENTIAL
        ):
            return loader_sampling
        return ds_sampling
```

Exactly two of the thirteen loaders declare a non-sequential preference:
`dataset/loader/random_pool.py:221` returns `DatasetSamplingStrategy.SHUFFLE` and
`dataset/loader/dag_jsonl.py:183` returns `DatasetSamplingStrategy.RANDOM`. Both
require an explicit `--custom-dataset-type` plus `--input-file`
(`config/flags/resolver.py:712` enforces the `random_pool` pairing), so the
default synthetic path is unaffected.

**Rust evidence** — `rust/cli/src/load.rs:1026-1029` unconditionally materializes
`"sequential"` when the flag is omitted:

```rust
                        .unwrap_or_else(|| "sequential".to_string()),
```

(the same expression appears at `rust/cli/src/load.rs:522-525` for the primary
dataset). No loader-preference override exists anywhere in
`rust/runtime/src/config/`: a search for `Shuffle`/`"shuffle"` across
`rust/runtime/src/` finds only the sampler implementation and its registration
(`rust/runtime/src/dataset/sampler.rs:94-102`, `:131`) plus test fixtures, never
a default-selection site. The value is consumed verbatim, e.g.
`rust/runtime/src/config/resolve.rs:1813`:

```rust
        recorded.shuffle = inputs.sampling != "sequential";
```

**Observable user impact:** With `--custom-dataset-type dag_jsonl`, Python samples
traces randomly with replacement while Rust walks them in authored order; with
`--custom-dataset-type random_pool`, Python shuffles and iterates without
replacement (re-shuffling on exhaustion) while Rust walks in order. Same command,
different prompts at each request index, and — because ordering determines prefix
overlap — a materially different KV-cache hit rate and therefore different TTFT
and throughput. Note that Python's override keys off the *value*, not whether the
user set it, so a user who explicitly passes
`--dataset-sampling-strategy sequential` against `dag_jsonl` still gets `random`
in Python while Rust honors the request; the divergence exists in both the
omitted and the explicitly-sequential case.

**Confidence:** High.

### 12. `--transport` / `--transport-type` exist on `origin/main` and are not accepted by Rust

**Severity:** P2
**Status:** NEW (invisible before the baseline correction)

**Python evidence** — `config/flags/cli_config.py:339-347`:

```python
    transport: Annotated[
        TransportType | None,
        Field(
            description="Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme "
            "(`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, "
            "TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.",
        ),
        CLIParameter(
            name=("--transport", "--transport-type"),
```

**Rust evidence** — no counterpart. `rg 'transport' rust/cli/src/flags.rs`
returns only `--dry-run`'s doc comment (`rust/cli/src/flags.rs:1377-1380`,
"Sets `transport.type: dry_run`"). These are the only 2 of the baseline's 301
long names/aliases that Rust does not accept.

**Observable user impact:** `aiperf profile --transport http ...` fails with a
clap "unexpected argument" error rather than running. This is a loud refusal, not
a silent change, and is therefore rated P2 per the audit's scope rule — but the
refusal is undocumented and Python accepted the flag, so a stored command line or
wrapper script that pins the transport explicitly breaks on upgrade with no
migration note. Real-world impact is limited because the flag's only supported
value (`http`) is what URL-scheme auto-detection already picks, and Python's own
help says the "explicit override [is] rarely needed."

**Confidence:** High.

### 13. `--request-cancellation-delay` without a rate: Python refuses, Rust silently drops

**Severity:** P2
**Status:** STILL VALID

**Python evidence** — `config/flags/_converter_profiling.py:561-575` raises,
with a comment naming silent dropping as the thing being avoided:

```python
    elif delay_set:
        # Mirror --arrival-smoothness gating: refuse to silently drop a
        # user-supplied flag whose dependency wasn't met.
        raise ValueError(
            "--request-cancellation-delay requires --request-cancellation-rate "
            "to be set (cancellation is disabled when rate is unset). "
```

The gate is on `model_fields_set` (`:561`), so it fires only for an explicitly
passed delay, not for the `0.0` default (`config/flags/cli_config.py:2668`).

**Rust evidence** — `rust/cli/src/load.rs:546-552` discards the delay through the
catch-all arm:

```rust
                        cancellation: match (
                            flags.request_cancellation_rate,
                            flags.request_cancellation_delay,
                        ) {
                            (Some(rate), delay) => Some((rate, delay.unwrap_or(0.0))),
                            _ => None,
                        },
```

**Observable user impact:** A user who omits or typos
`--request-cancellation-rate` gets an actionable error from Python and a clean
successful run with zero cancellations from Rust. The run *looks* like it
exercised cancellation. Impact is bounded because the intended feature was off in
both cases; the harm is false confidence, hence P2.

**Confidence:** High.

### 14. `--num-sessions` and `--num-conversations` are one field in Python and two in Rust, changing which wins

**Severity:** P2
**Status:** STILL VALID

**Python evidence** — `config/flags/cli_config.py:901-923`: three names, one
field (`conversation_num`), so ordinary last-one-wins argv semantics apply:

```python
            name=(
                "--conversation-num",
                "--num-conversations",
                "--num-sessions",
            ),
```

**Rust evidence** — two independent fields, `rust/cli/src/flags.rs:56`
(`--num-conversations`, `visible_alias = "conversation-num"`) and
`rust/cli/src/flags.rs:620-622` (`--num-sessions`), resolved by fixed precedence
rather than argv order at `rust/cli/src/load.rs:527-533`:

```rust
                    entries: num_dataset_entries
                        .or(num_conversations)
                        .or(num_sessions)
                        ...
                    sessions: num_conversations.or(num_sessions).map(u64::from),
```

**Observable user impact:** `--num-sessions 100 --num-conversations 5` runs 100
sessions in Python and 5 in Rust; reversing the argv order changes Python's answer
to 5 but leaves Rust at 5. Any generated command line or config-templating layer
that emits both names — plausible precisely because Python advertised them as
synonyms — silently benchmarks a 20×-different session count. P2 because passing
both spellings of one option is uncommon in hand-written commands.

**Confidence:** High.

## Withdrawn after baseline correction

No finding was withdrawn outright; every divergence reported against the branch
tree also exists against `origin/main`. Three sub-claims were wrong and have been
corrected in place, and two unverified items were branch artifacts:

- **Finding 2 (audio sample rates) — sub-claim corrected.** I originally reported
  an unconditional 1000× divergence in both directions. Baseline has a
  threshold-based normalizer (`config/flags/_converter_dataset.py:28-38`,
  `v / 1000.0 if v > 96.0 else v`) that the branch tree also had but which I did
  not find. The two implementations therefore *agree* for Hz-shaped values above
  96; the divergence is confined to values at or below 96. The finding survives
  with a narrower, correctly-shaped claim.
- **Finding 11 (preferred sampling) — blast radius corrected.** I originally
  implied this affected all synthetic runs. `random_pool` requires an explicit
  `--custom-dataset-type random_pool` plus `--input-file`
  (`config/flags/resolver.py:712`), so the scope is two opt-in dataset types
  (`random_pool`, `dag_jsonl`), not the default synthetic path.
- **Finding 7 (stddev) — OSL half demoted.** I originally asserted both the ISL
  and OSL halves as divergences. Baseline's own help text
  (`config/flags/cli_config.py:1499`) says `--osl-stddev` "Only applies when
  `--prompt-output-tokens-mean` is set", which may make Rust's behavior correct
  for OSL. The ISL half is unaffected and remains asserted; the OSL half moved to
  Unverified.
- **Unverified item "`--scenario` interaction with the 10-request fallback" —
  withdrawn as a branch artifact.** The `and cli.scenario is None` clause I cited,
  along with its long explanatory comment about agentic scenarios, exists only in
  the branch's local edits. Baseline's fallback
  (`config/flags/_converter_profiling.py:552-554`) excludes only
  `FIXED_SCHEDULE`, and Rust's `effective_requests`
  (`rust/runtime/src/config/resolve.rs:1353-1359`) likewise has no scenario
  clause. There is no divergence here.
- **Unverified item "trajectory ratio defaults" — resolved into Finding 5.** The
  branch tree declared these `float | None` with `default=None`, which is why the
  0.25/0.75 values could not be confirmed. Baseline declares them as plain
  `float` with `] = 0.25` and `] = 0.75`
  (`config/flags/cli_config.py:2312`, `:2329`), making this a definite P1.

## Checked and consistent

Verified as matching against baseline `bc359bf8fd`; no need to re-check.

- **Flag name coverage.** 299 of the 301 long names/aliases published by
  `config/flags/cli_config.py` are accepted by the Rust CLI (522 long names
  total, extracted across `rust/cli/src/**/*.rs` including `long = "..."`,
  `alias`/`visible_alias`, `aliases`/`visible_aliases`, and bare `#[arg(long)]`
  field-derived names). The only two absent are `--transport` and
  `--transport-type` (Finding 12). In particular `--max-workers`,
  `--profile-export-level`, `--record-processors`, `--sequence-distribution`,
  `--sweep-variant`, and `--ui` are all present as Rust `visible_alias`es
  (`rust/cli/src/flags.rs:100`, `:156`, `:493`, `:539`, `:542`, `:547`).
- **Short flags.** Baseline publishes `-H -b -f -m -u -v -vv`; Rust provides
  `-H -b -f -m -u -v` plus `-vv` as a `visible_alias`
  (`rust/cli/src/flags.rs:533`). Complete coverage.
- **No per-flag environment variables on either side.** Baseline's
  `config/cli_parameter.py:24` forces `show_env_var=False` and no `env_var=` is
  set on any parameter; `App(name="aiperf", ...)` at `cli.py:24` sets no prefix.
  Rust's `rust/cli/src/flags.rs` contains no `env = "..."` attributes at all. The
  env-vs-CLI precedence question is therefore vacuous for this surface. (Rust's
  `AIPERF_*` runtime knobs such as `AIPERF_METRICS_SKETCH` are Rust-only
  additions, out of scope.)
- **Default request bound = 10 requests.** Python
  `config/flags/_converter_profiling.py:552-560` and Rust `DEFAULT_REQUEST_COUNT
  = 10` (`rust/runtime/src/config/resolve.rs:80`) agree for the
  non-user-centric case. Note `config/flags/cli_config.py:2212` still claims
  `max(10, concurrency * 2)`; that is stale documentation *on the Python side*,
  not a parity gap — the baseline code does not implement it.
- **`--benchmark-grace-period`** default 30.0s both
  (`rust/runtime/src/config/phase_validate.rs:13`), and both apply it only when a
  benchmark duration is set (`rust/runtime/src/config/resolve.rs:1360-1364`).
- **`--request-timeout-seconds`** 21600.0 (6h) both
  (`rust/runtime/src/config/resolve.rs:75`).
- **`--wait-for-model-timeout`** 0.0 both; **`--wait-for-model-interval`** 5.0
  both (`rust/runtime/src/config/resolve.rs:78`).
- **`--request-cancellation-rate`** is a percentage (0–100) on both sides;
  `rust/runtime/src/ancillary.rs` names the field `cancellation_rate_percent`.
  The suspected percent-vs-fraction mismatch does not exist.
- **`--batch-size`** (aliases `--batch-size-text`, `--prompt-batch-size`, short
  `-b`) defaults to 1 on both sides (`rust/cli/src/load.rs:521`).
- **`--isl` mean default 550** (`config/flags/cli_config.py:1359`;
  Rust `DEFAULT_ISL_MEAN = 550.0`) and **`--osl` default unset**
  (`config/flags/cli_config.py:1491`; Rust `Option`).
- **`--video-audio-num-channels`** is spelled identically on both sides; the
  baseline Python *field* is `video_audio_channels` but its `CLIParameter` name
  is `--video-audio-num-channels`, and both default to 0
  (`config/flags/cli_config.py:1889`).
- **Rankings stddev semantics**: stddev defaults to 0.0 alongside a supplied mean
  on both sides (`rust/cli/src/load.rs:1100-1112`).
- **Video defaults other than duration**: `--video-fps 4`, `--video-format webm`,
  `--video-codec libvpx-vp9`, `--video-synth-type moving_shapes`,
  `--video-audio-depth 16` all match.
- **Audio defaults when the flag is omitted**: sample rate 16 kHz, depth 16,
  format `wav`, 1 channel — all match (Finding 2 is confined to
  explicitly-supplied values).
- **`UNIMPLEMENTED_FLAGS` are not silent.** The 23 entries at
  `rust/cli/src/profile.rs:383-417` (`--api-host`, `--api-port`, `--ui-type`,
  `--workers-max`, `--stats-interval`, `--record-processor-service-count`, the
  `--zmq-*` set, the search/convergence family, `--plot-required`) each emit a
  `tracing::warn!` naming the flag. Out of scope as loud refusals — and this
  retires most of the specific examples in the pre-existing P1.4. `--auto-plot`
  is the one gap (Finding 4).
- **`--num-profile-runs` (1–10) and `--confidence-level` (0 < c < 1)** are
  explicitly validated in `rust/cli/src/profile.rs:435-479`; bounds are enforced,
  not clamped.
- **`--fixed-schedule` request bound** is derived from the input file's non-empty
  entry count on both sides (`config/flags/_converter_profiling.py:659-667`;
  `rust/cli/src/load.rs:301-306`).
- **Recorded-agent graph flags** are loudly rejected by Rust when
  `--graph-format` is not `agent_recording`
  (`reject_inapplicable_recorded_agent_flags`), not silently ignored.

## Unverified / needs runtime check

- **`--osl-stddev` without `--osl`.** Baseline's converter forwards
  `osl["stddev"]` with no mean (`config/flags/_converter_dataset.py:72-79`), but
  the flag's own help says it "Only applies when `--prompt-output-tokens-mean` is
  set" (`config/flags/cli_config.py:1499`). If the help is accurate about
  downstream behavior, Rust's total drop
  (`rust/cli/src/load.rs:481-485`) is correct and there is no finding; if the
  converter's forwarding is honored downstream, this is a second instance of
  Finding 7. Needs the Python OSL consumption site traced, or one run with
  `--osl-stddev 64` alone inspecting per-record requested `max_completion_tokens`.
- **Finding 10's exact magnitudes.** The rankings token counts and the audio
  `default_media_dim()` value of 512 are read from the resolution layer only. The
  units of `audio_length_mean` (seconds?) and whether a fixed `value` distribution
  is honored identically to a `{mean, stddev}` one are not confirmed. Needs one
  `--rankings-passages-mean 5` run per side comparing per-record ISL, and one
  `--audio-batch-size 2` run comparing emitted audio length.
- **Whether Rust's Poisson phase silently accepts or rejects a smoothness value
  elsewhere.** Finding 3 establishes that `PhaseKind::Poisson` has no smoothness
  field so the value cannot be carried, but I did not check whether any Config-v2
  validator rejects `smoothness` on a Poisson phase (which would make it a loud
  refusal rather than a silent drop for the YAML path — the CLI path still
  silently drops it, since `rate_mode` is `None`). Needs
  `rust/runtime/src/config/phase_validate.rs` read for a smoothness/kind
  cross-check.
- **`--dataset-sampling-strategy` value-space strictness.** Python types the flag
  as a `DatasetSamplingStrategy` enum (`config/flags/cli_config.py:725-726`) so an
  unknown value is rejected at parse time; Rust threads a bare `String`
  (`rust/runtime/src/config/resolve.rs:375`, `pub sampling: String`). Whether an
  unknown strategy name fails closed at sampler-factory lookup
  (`rust/runtime/src/dataset/sampler.rs:131`) or silently falls back was not
  traced. If it falls back, that is a further finding in this family.
- **The `--plot` → `auto_plot` implication.** Baseline
  `config/config.py:1061` (`_plot_implies_auto_plot`) flips `auto_plot` to True
  when `--plot` is passed. Given Finding 4 (Rust never reads `auto_plot` at all),
  this is almost certainly subsumed, but I did not check whether Rust has a
  `--plot` flag with its own behavior.
