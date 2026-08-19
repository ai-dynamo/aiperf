---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Multi-Phase Workflows
---
# Multi-Phase Workflows

Use named multi-phase workflows when one benchmark run needs several ordered load windows, such as cache warmup, baseline profiling, a cancellation storm, recovery, and a final profiling window.

Each phase has two identities:

| Field | Meaning |
| --- | --- |
| `name` | Unique workflow identifier. It is used in sweep paths, logs, and artifact directories. |
| `kind` | Semantic role: `warmup` or `profiling`. It controls whether the phase contributes to benchmark results. |

Canonical names infer their kind. A phase named `warmup` defaults to `kind: warmup`, and a phase named `profiling` defaults to `kind: profiling`. Custom names must set `kind` explicitly.

## Example

This workflow warms the service, records a low-cancellation baseline, runs a storm window, allows recovery, and records a final steady window:

```yaml
schemaVersion: "2.0"

benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 1000
    prompts: {isl: 512, osl: 128}
  phases:
    - name: warmup
      type: concurrency
      concurrency: 8
      duration: 5m

    - name: baseline_traffic
      kind: profiling
      type: concurrency
      concurrency: 32
      duration: 30m
      cancellation: {rate: 5, delay: 0}

    - name: cancellation_stress
      kind: profiling
      type: concurrency
      concurrency: 64
      duration: 5m
      cancellation: {rate: 50, delay: 0}

    - name: recovery_traffic
      kind: profiling
      type: concurrency
      concurrency: 32
      duration: 30m
      cancellation: {rate: 0, delay: 0}
  artifacts:
    dir: ./artifacts/multi-phase
```

Run it with:

```bash
aiperf profile --config multi-phase.yaml --ui none
```

## Multiple Warmups

You can have more than one warmup phase. Only one phase can literally be named `warmup`, because phase names must be unique. Give additional warmup windows custom names and set `kind: warmup`:

```yaml
benchmark:
  phases:
    - name: warmup
      type: concurrency
      concurrency: 4
      requests: 50

    - name: cache_prime
      kind: warmup
      type: poisson
      rate: 20
      duration: 2m

    - name: profiling
      type: concurrency
      concurrency: 16
      duration: 10m
```

Warmup phases are always excluded from aggregate profiling results. They can still have their own phase-scoped artifacts, which is useful when you want to inspect how setup traffic behaved.

A config must include at least one `profiling` phase. Warmup-only workflows are rejected.

## Naming Rules

Phase names must be strict identifiers: `^[A-Za-z_][A-Za-z0-9_-]*$`. Names are unique case-insensitively because they become artifact directory names and sweep path components.

The canonical names `warmup` and `profiling` are reserved for their matching kind. For example, `name: warmup` cannot be paired with `kind: profiling`.

Use custom names when the workflow has repeated semantic phases:

```yaml
phases:
  - {name: settle_before_stress, kind: warmup, type: concurrency, duration: 5m}
  - {name: cancellation_stress, kind: profiling, type: concurrency, duration: 5m}
  - {name: settle_after_stress, kind: warmup, type: concurrency, duration: 5m}
  - {name: final_profile, kind: profiling, type: concurrency, duration: 10m}
```

## Artifacts

AIPerf keeps the root exports for backward compatibility. The metrics files are the traditional profile outputs; observability files are present when GPU telemetry or server metrics are enabled and produce data:

```text
profile_export_aiperf.json
profile_export_aiperf.csv
server_metrics_export.json
server_metrics_export.csv
server_metrics_export.jsonl
gpu_telemetry_export.jsonl
```

Named phases add a manifest and phase-scoped files under `phases/<phase_name>/`:

```text
phase_manifest.json
phases/warmup/profile_export_aiperf.json
phases/warmup/profile_export_aiperf.csv
phases/baseline_traffic/profile_export_aiperf.json
phases/baseline_traffic/profile_export_aiperf.csv
phases/baseline_traffic/server_metrics.json
phases/baseline_traffic/gpu_telemetry.json
```

`phase_manifest.json` is the discovery file. Each entry includes phase identity fields such as `phase_index`, `profiling_index`, `phase_name`, and `phase_kind`, plus artifact paths such as `metrics_json`, `metrics_csv`, `server_metrics_json`, and `gpu_telemetry_json` when those files were written.

Per-phase `server_metrics.json` and `gpu_telemetry.json` are best-effort observability exports. If server metrics or GPU telemetry is disabled, unavailable, or produced no data for a phase, the corresponding phase file is omitted and the manifest entry does not include that path. Metrics JSON/CSV files are still written for the phase when the normal metric exporters are enabled.

## Aggregate vs Phase-Scoped Results

Use the root `profile_export_aiperf.json` and CSV when you need the traditional benchmark output shape for existing tools. Warmup phases are excluded from those aggregate profiling results.

Use `phase_manifest.json` and `phases/<phase_name>/profile_export_aiperf.json` when you need exact per-phase request metrics. This is the right view for comparing `baseline_traffic`, `cancellation_stress`, and `recovery_traffic` independently.

For server metrics, the root `server_metrics_export.json` preserves the legacy aggregate view. In workflows with repeated phase kinds, its phase ranges may span gaps between phases of the same kind. Use `phases/<phase_name>/server_metrics.json` for exact phase-scoped server telemetry.

Adaptive profiling phases also write their own controller artifacts. See [Adaptive Scale](adaptive-scale.md#multiple-adaptive-phases) for multiple adaptive phases and `adaptive_scale_manifest.json`.

## CLI Overrides and Sweeps

With exactly one profiling phase, normal CLI loadgen flags can still target that phase. With multiple profiling phases, put phase-specific load settings directly in YAML so the target is unambiguous.

Sweeps can target named phase fields through the phase name:

```yaml
sweep:
  type: grid
  parameters:
    phases.baseline_traffic.concurrency: [16, 32]
    phases.cancellation_stress.cancellation.rate: [25, 50]
```

## Seamless Transitions

By default, AIPerf waits for a phase's in-flight requests to finish before moving to the next phase. Set `seamless: true` on a later phase when the next phase should start immediately after the previous phase stops issuing new work:

```yaml
phases:
  - name: baseline_traffic
    kind: profiling
    type: concurrency
    duration: 30m

  - name: cancellation_stress
    kind: profiling
    type: concurrency
    duration: 5m
    seamless: true
```

`seamless` cannot be set on the first phase because there is no previous phase to transition from.
