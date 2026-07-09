---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Adaptive Scale
---
# Adaptive Scale

Adaptive scale is a single-run load controller for finding and sustaining an SLA boundary. Instead of launching many independent sweep or search runs, AIPerf starts at a low control value, evaluates SLA windows, increases load while every SLA filter passes, and then sustains near the last passing boundary.

Use adaptive scale when you want one benchmark invocation to push a service until a latency, reliability, or goodput constraint starts failing, then keep pressure near that edge. Use `adaptive_search` when you want offline Bayesian optimization across multiple runs, sweeps when you want a fixed experiment grid, fixed ramps when you already know the schedule you want to replay, and static concurrency or request rate when you only need one fixed load point.

## YAML example

YAML is the preferred way to configure adaptive scale because it keeps the control variable, assessment windows, sustain period, and SLA filters together. The canonical shape uses a nested `adaptive_scale.control` block:

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
        ttft:
          p95:
            le: 3000
        inter_token_latency:
          p95:
            le: 100
        goodput:
          avg:
            ge: 20
        error_rate:
          avg:
            le: 0.01
```

A window passes only when every SLA filter passes. Latency-family thresholds use milliseconds. Lower-is-better metrics usually use `lt` or `le`; higher-is-better metrics such as throughput, goodput, goodput ratio, and success rate usually use `gt` or `ge`.

Do not combine adaptive scale with a fixed ramp on the same variable. For example, `control.variable: prefill_concurrency` cannot be used with `prefill_ramp`, and `control.variable: request_rate` cannot be used with `rate_ramp`.

## Control variables

| Control variable | Phase shape | What changes |
| --- | --- | --- |
| `concurrency` | `type: concurrency` or another phase with a concurrency ceiling | In-flight session concurrency. |
| `prefill_concurrency` | Streaming phases with `concurrency` and `prefill_concurrency` | Simultaneous prefill-heavy requests, while total concurrency remains capped. |
| `request_rate` | Rate-controlled phases such as `poisson`, `constant`, or `gamma` | Scheduled request rate. `concurrency` still acts as an in-flight ceiling when set. |
| `users` | `type: user_centric` | Live simulated user timelines. Total target QPS remains fixed, so per-user turn gap changes. |

For `users`, adaptive scale changes population pressure rather than acting as another spelling of request rate.

## CLI quick start

The CLI is useful for scripts and simple one-phase runs. Prefer YAML for reusable benchmark definitions.

```bash
aiperf profile \
  --url http://localhost:8000/v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint-type chat \
  --streaming \
  --concurrency 400 \
  --benchmark-duration 3600 \
  --adaptive-scale \
  --adaptive-scale-control concurrency:1,400:int \
  --adaptive-scale-assessment-period 60 \
  --adaptive-sustain-duration 1800 \
  --adaptive-scale-sla request_latency:p95:le:30000 \
  --adaptive-scale-sla error_rate:avg:le:0.01
```

The compact control form is `--adaptive-scale-control variable:min,max:type`. Use `int` for `concurrency`, `prefill_concurrency`, and `users`; use `float` for `request_rate`. Do not mix compact control with expanded `--adaptive-control-*` flags.

## Artifacts

Adaptive scale writes these timing-owned artifacts into the run artifact directory:

```text
adaptive_scale_events.jsonl
adaptive_scale_summary.json
```

`adaptive_scale_events.jsonl` is an event stream for orchestration and post-processing. Each line includes `schema_version`, timestamps, `event`, `control_variable`, `control_value_before`, `control_value_after`, `boundary_value`, `last_passing_value`, `first_failing_value`, `sla_values`, and `binding_sla` fields. Pollers should key off explicit events such as `sustain_started` rather than sleeping for a fixed amount of time.

`adaptive_scale_summary.json` is the final controller summary. It records the discovered boundary, final control value, last passing value, first failing value, sustain status, throughput, sample counts, error counts, cancellation counts, and the evaluated candidate windows.

Artifact fields are intended for orchestration-facing consumers, so treat schema changes and field renames as compatibility events.

## Metric semantics

TTFT and ITL SLA samples come from successful completed requests only. Reliability is represented separately by `success_rate`, `error_rate`, and `cancellation_rate`.

`goodput` is the throughput of successful requests that pass the configured per-request quality filters. `goodput_ratio` is quality-passing successful requests divided by total attempts. `success_rate` is successful completed requests divided by total attempts.

See [Adaptive SLA metric support](yaml-config.md#adaptive-sla-metric-support) for the full metric and statistic matrix.
