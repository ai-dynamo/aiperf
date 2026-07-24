<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive scale

## Purpose

Adaptive scale is a closed-loop SLA controller layered over an already-running
load phase. It does not issue credits; it keeps the existing
request-rate/concurrency issuance path running and, on a Clock-paced background
assessment task, samples SLA metrics per window, decides pass/fail, and mutates
one control knob to ramp load up until the SLA breaks, then holds at the
last-good level.

## Built

`aiperf_runtime::adaptive_core` provides object-safe actuator, evaluator, step,
window, and controller seams. Online and offline modes share the injected-backend
futures.

### Control

The one strategy is `ramp_until_fail`: a monotone upward ramp with an
SLA-margin-scaled step, a boundary-discovery step-back, and a single-recovery
sustain hold, across three controller phases `discover → sustain → complete`. It
is not a binary or Bayesian search (that is the separate search planner, see
[metrics.md](metrics.md) for measured inputs).

The control knob is abstracted behind an actuator; the four live actuators —
session concurrency, prefill concurrency, request rate, and target users — map
onto the built ramp actuators `SlotPool::set_limit` and
`IntervalGenerator::set_rate`. SLA math reconciles authoritative
completion-token OSL/ITL from server usage. Assessment windows are Clock-paced.

### Artifacts

The controller emits schema-v2 adaptive events and a summary artifact. Two
concurrent time-driven activities run during an adaptive phase: the unchanged
credit issuer and the background assessment loop that resizes the pools or rate
the issuer obeys.

## Source anchors

- `rust/runtime/src/adaptive_core/` (`actuator.rs`, `controller.rs`, `sla.rs`,
  `step.rs`, `window.rs`, `artifacts.rs`, `observer.rs`, `runtime.rs`).
- `rust/runtime/src/run.rs` (online/offline composition).
- `rust/runtime/src/timing/{intervals.rs,slots.rs,stop.rs}` (ramp actuators).
