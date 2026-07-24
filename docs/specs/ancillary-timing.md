<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Ancillary timing policy

## Purpose

Three ancillary timing knobs ride on top of the credit-issuer loop
(see [scheduling.md](scheduling.md)). None is a workload of its own; each
perturbs an already-running phase over the injected `Clock`.

## Built

`aiperf_runtime::timing` owns all three, consumed by both the HTTP path and the
in-process offline runtime.

### Ramping

Smoothly walk the target rate (`IntervalGenerator::set_rate`) or concurrency
limit (`SlotPool::set_limit`) from a start value to a target over a fixed
duration on a pluggable curve — Linear, Exponential, or Poisson. Ramps are
Clock-driven. Fixed schedules reject ramps.

### Request cancellation

Seeded, warmup-aware probabilistic client-disconnect timers arm on a fraction of
requests to exercise mid-response disconnect handling. On HTTP the cancel timer
starts at body-send completion (not at issuance); the offline path calls the
steppable engine's terminal operation. The single in-process endpoint rejects URL
selection.

### URL selection

When several endpoint URLs are given, sticky round-robin selection picks which
endpoint each conversation hits. The round-robin advances on turn 0 only, then
pins per session.

### Backend-neutral consumption

Paced, request-rate, and user-centric paths consume the applicable controls over
either the online or offline clock/dispatcher pair. Graph phases consume arrival
and prefill ramps and adaptive concurrency/prefill/request-rate controls through
placement-wide worker updates (see [graph-runtime.md](graph-runtime.md)).

## Source anchors

- `rust/runtime/src/timing/{ramping.rs,cancellation.rs,url_selection.rs,arrival.rs,intervals.rs,slots.rs}`.
