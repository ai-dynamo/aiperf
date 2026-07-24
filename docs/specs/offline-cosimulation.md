<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Offline co-simulation

## Purpose

Socket-free Dynamo co-simulation: AIPerf drives the Dynamo mocker's inference
engine in-process, feeding AIPerf's own measurement stream so every command
produces AIPerf's report offline as well as online. It is enabled by the
`dynosim` Cargo feature and reached through Config v2
(`transport.type: dynosim_offline` for virtual time, `dynosim_online` for
wall-clock in-process replay). Default builds omit the transports.

## Built

### Steppable engine boundary

`aiperf_runtime::dynosim` inverts the mocker's self-driving batch loops into a
passive, steppable, externally clocked engine core. The engine takes `now` as a
plain scalar and a `&mut dyn RequestObserver`; it never sees a clock object.
`SteppableEngine` (and `SteppableAgg`/`SteppableDisagg` hosts) expose
`step_to(now_ms, observer)` plus `next_event_ms`, so AIPerf owns the run loop and
the clock. Dependency is strictly `aiperf → mocker`: the mocker consumed from the
sibling checkout under the `dynosim` feature never depends on AIPerf or its
`Clock`. The scheduler math is untouched; only the driver loops are inverted.

`SimClock` supplies integer-nanosecond virtual time (`dynosim_offline`);
`RealClock` supplies wall time (`dynosim_online`). One driver, three modes
(online-real, online-mock, offline-mock), one report type. The steppable path
reproduces the batch path's `perf_ns` sequence bit-for-bit on every handoff
conformance fixture — the acceptance gate. AIPerf emits its normal native-v2
report; no report-schema change.

### Level-B observer contract

Offline co-simulation feeds AIPerf's own `RequestObserver`/collector rather than
an internal dump. The engine's pass emits through `&mut dyn RequestObserver`
instead of a concrete collector; the first `on_token` observation is the first
token, releasing the prefill slot and firing the graph first-token gate.
AIPerf's collector is primary on every backend; the mocker's `TraceCollector` is
an optional co-observer tee'd alongside it. Because per-token events fire during
the run, adaptive-scale windows, streaming metrics, and the live dashboard work
offline, not just as a post-hoc dump.

## Source anchors

- `rust/runtime/src/dynosim.rs` (`SteppableEngine`, `step_to`/`next_event_ms`,
  `RequestObserver`-driven dispatch), `rust/runtime/src/aic_runtime.rs`.
- `rust/runtime/src/engine/offline_execution.rs`,
  `rust/runtime/src/clock/sim_clock.rs`.
- `dynosim` feature builds only; requires the sibling `dynamo-aiperf-native`
  checkout.
