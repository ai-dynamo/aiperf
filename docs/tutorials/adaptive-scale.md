---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Adaptive Scale
---
# Adaptive Scale

Adaptive scale is a single-run load controller for finding and sustaining an SLA boundary. Instead of launching many independent sweep or search runs, AIPerf starts at a low control value, evaluates SLA windows, increases load while every SLA filter passes, and then sustains near the last passing boundary.

Use adaptive scale when you want one benchmark invocation to push a service until a latency or throughput constraint starts failing, then keep pressure near that edge. Use `adaptive_search` when you want offline Bayesian optimization across multiple runs, sweeps when you want a fixed experiment grid, fixed ramps when you already know the schedule you want to replay, and static concurrency or request rate when you only need one fixed load point.

Adaptive scale is only configurable in YAML. Put the adaptive settings in the profiling phase's `adaptive_scale` block, define the phase-level `sla` filters, then run the benchmark with `aiperf profile --config <file>`.

## YAML example

Adaptive scale is configured in YAML because it keeps the control variable, assessment windows, sustain period, and SLA filters together with the phase they control. The examples below set `kind: profiling` explicitly to show the current phase model, but existing configs that use canonical `warmup`/`profiling` names without `kind` remain valid because the loader infers it. The canonical shape uses a nested `adaptive_scale.control` block:

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
    - name: profiling
      kind: profiling
      type: concurrency
      concurrency: 200
      prefill_concurrency: 64
      duration: 3600
      adaptive_scale:
        enabled: true
        control:
          variable: prefill_concurrency
          min: 1
          max: 64
        assessment_period: 60
        min_completed_requests: 20
        sustain_duration: 1800
        strategy:
          type: ramp_until_fail
          step_policy: sla_margin
          base_step: 10
          max_step_multiplier: 4
      sla:
        request_latency:
          p95:
            le: 30000
        request_throughput:
          avg:
            ge: 1
```

A window passes only when every SLA filter passes. Latency, TTFT, and ITL thresholds use milliseconds. Lower-is-better metrics usually use `lt` or `le`; higher-is-better metrics such as request throughput, success rate, goodput, and goodput ratio usually use `gt` or `ge`.

The canonical adaptive shape uses `adaptive_scale.control` for the load-control variable and phase-level `sla` for the filters. `duration`, `adaptive_scale.sustain_duration`, and at least one SLA filter are required when adaptive scale is enabled. Existing flat concurrency aliases such as `min_concurrency` and `max_concurrency`, plus `adaptive_scale.sla`, are still accepted for compatibility, but new configs should use `control.min`, `control.max`, and phase-level `sla`.

Do not combine adaptive scale with a fixed ramp on the same variable. For example, `control.variable: prefill_concurrency` cannot be used with `prefill_ramp`, and `control.variable: request_rate` cannot be used with `rate_ramp`. Fixed ramps for other variables are allowed.

## Boundary Example

This example demonstrates adaptive scale discovering a latency boundary against an OpenAI-compatible chat endpoint. Make sure the endpoint is reachable at `localhost:8000`, then tune the `model` and SLA threshold for your service.

```yaml
schemaVersion: "2.0"

benchmark:
  model: Qwen/Qwen3-0.6B
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 1000
    prompts:
      isl: 128
      osl: 16
  phases:
    - name: profiling
      kind: profiling
      type: concurrency
      concurrency: 4
      duration: 45
      adaptive_scale:
        enabled: true
        control:
          variable: concurrency
          min: 4
          max: 64
        assessment_period: 5
        min_completed_requests: 5
        sustain_duration: 10
        strategy:
          type: ramp_until_fail
          step_policy: fixed_percent_step
          step_percent: 100
      sla:
        request_latency:
          p95:
            le: 80
```

Run it with:

```bash
aiperf profile \
  --config adaptive-scale-boundary.yaml \
  --tokenizer builtin \
  --extra-inputs ignore_eos:true \
  --ui none \
  --output-artifact-dir ./adaptive-scale-boundary-artifacts
```

**Sample Output (Successful Run):**

```text
concurrency 4  -> request_latency:p95 = 54.1 ms, pass
concurrency 8  -> request_latency:p95 = 58.7 ms, pass
concurrency 16 -> request_latency:p95 = 56.1 ms, pass
concurrency 32 -> request_latency:p95 = 73.7 ms, pass
concurrency 64 -> request_latency:p95 = 98.2 ms, fail
sustain at 32  -> request_latency:p95 = 72.0 ms, then 74.4 ms, pass
```

The corresponding `adaptive_scale_summary.json` included:

```json
{
  "status": "completed",
  "completed_reason": "sustain_duration_completed",
  "control_variable": "concurrency",
  "control_value": 32,
  "boundary_value": 32,
  "last_passing_value": 32,
  "first_failing_value": 64,
  "sustain_windows": 2,
  "sustain_passed_windows": 2
}
```

Exact numbers depend on your server and hardware. If the run passes at `max`, lower the latency threshold or raise `adaptive_scale.control.max`. If it fails at `min`, loosen the latency threshold or lower `adaptive_scale.control.min`.

## Control variables

| Control variable | Phase shape | What changes |
| --- | --- | --- |
| `concurrency` | `type: concurrency` or another phase with a concurrency ceiling | In-flight session concurrency. |
| `prefill_concurrency` | Streaming phases with `concurrency` and `prefill_concurrency` | Simultaneous prefill-heavy requests, while total concurrency remains capped. |
| `request_rate` | Rate-controlled phases such as `poisson`, `constant`, or `gamma` | Scheduled request rate. `concurrency` still acts as an in-flight ceiling when set. |
| `users` | `type: user_centric` | Live simulated user timelines. Total target QPS remains fixed, so per-user turn gap changes. |

For `users`, adaptive scale changes population pressure rather than acting as another spelling of request rate.

## SLA filters

Adaptive scale supports these SLA metric families in this release:

| Metric family | Metric tags and aliases | Supported stats |
| --- | --- | --- |
| E2E latency | `request_latency` | `avg`, `min`, `max`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` |
| Time to first token | `time_to_first_token`, `ttft` | `avg`, `min`, `max`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` |
| Inter-token latency | `inter_token_latency`, `itl`, `tpot` | `avg`, `min`, `max`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` |
| Request throughput | `throughput`, `request_throughput`, `completed_request_throughput` | `avg`, `min`, `max` |
| Output token throughput | `output_token_throughput` | `avg`, `min`, `max` |
| Quality goodput | `goodput` | `avg`, `min`, `max` |
| Goodput ratio | `goodput_ratio` | `avg`, `min`, `max` |
| Success rate | `success_rate`, `request_success_rate` | `avg`, `min`, `max` |
| Error rate | `error_rate`, `request_error_rate` | `avg`, `min`, `max` |
| Cancellation rate | `cancellation_rate`, `request_cancellation_rate` | `avg`, `min`, `max` |

Quality goodput and goodput ratio require at least one request-quality filter, such as `request_latency`, `time_to_first_token`, or `inter_token_latency`, so the controller can decide which successful requests count as quality-passing.

## YAML-only configuration

Adaptive scale does not expose standalone CLI flags. Put the adaptive settings in the target phase's `adaptive_scale` block, keep SLA filters on the phase-level `sla` block, and run the benchmark with `aiperf profile --config <file>`. General CLI flags such as `--tokenizer`, `--ui`, and `--output-artifact-dir` can still be used around the YAML benchmark definition.

## Multiple adaptive phases

A workflow can include more than one adaptive profiling phase. Give each phase a unique `name`, set `kind: profiling` for custom names, and put each phase's control bounds, assessment windows, sustain duration, strategy, and SLA filters directly in YAML.

```yaml
benchmark:
  phases:
    - name: baseline_traffic_search
      kind: profiling
      type: concurrency
      concurrency: 64
      duration: 30m
      adaptive_scale:
        enabled: true
        control: {variable: concurrency, min: 4, max: 64}
        assessment_period: 30
        sustain_duration: 5m
      sla:
        request_latency: {p95: {le: 300}}

    - name: cancellation_stress_search
      kind: profiling
      type: concurrency
      concurrency: 128
      duration: 30m
      cancellation: {rate: 25, delay: 0}
      adaptive_scale:
        enabled: true
        control: {variable: concurrency, min: 8, max: 128}
        assessment_period: 30
        sustain_duration: 5m
      sla:
        request_latency: {p95: {le: 500}}
        error_rate: {avg: {le: 0.05}}
```

Each adaptive phase runs its own controller and writes its own phase-scoped event and summary files. Use `adaptive_scale_manifest.json` to discover all adaptive phases in the run and locate their artifacts. For general `name` / `kind` rules and repeated warmup or profiling windows, see [Multi-Phase Workflows](multi-phase-workflows.md).

## Artifacts

Adaptive scale writes phase-scoped timing artifacts plus a manifest into the run artifact directory:

```text
phases/<phase_name>/adaptive_scale_events.jsonl
phases/<phase_name>/adaptive_scale_summary.json
adaptive_scale_manifest.json
```

A run with multiple adaptive profiling phases gets one event/summary pair per phase and one manifest entry per adaptive phase. Each manifest entry records the phase identity plus `events_path` and `summary_path` for that phase.

`adaptive_scale_events.jsonl` is an event stream for orchestration and post-processing. Each line includes `schema_version`, timestamps, `event`, `control_variable`, `control_value_before`, `control_value_after`, `boundary_value`, `last_passing_value`, `first_failing_value`, `sla_values`, and `binding_sla` fields. Pollers should key off explicit events such as `sustain_started` rather than sleeping for a fixed amount of time.

`adaptive_scale_summary.json` is the final controller summary for one adaptive phase. It records the discovered boundary, final control value, last passing value, first failing value, sustain status, throughput, sample counts, error counts, cancellation counts, and the evaluated candidate windows.

Artifact fields are intended for orchestration-facing consumers, so treat schema changes and field renames as compatibility events.

## Metric semantics

Use `request_latency` for simple latency-boundary examples. See [SLA filters](#sla-filters) for the supported adaptive SLA metric and statistic matrix.
