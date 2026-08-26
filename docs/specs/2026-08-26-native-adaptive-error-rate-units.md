<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native adaptive error-rate units

## Purpose

Keep native adaptive-scale error-rate SLAs semantically identical to the
exported `request_error_rate` metric while preserving the independent
success-rate and cancellation-rate contracts.

## Built contract

For `error_rate` and `request_error_rate`, all `avg`, `min`, and `max` filters
evaluate to `100 * errors / (successful requests + errors)`. A cancellation is
not in that denominator. A zero-completion window has value `0.0`; an all-error
window therefore evaluates to `100.0` and remains eligible for the existing
zero-success controller path.

Both aliases accept only finite thresholds in `[0, 100]`. A threshold strictly
between `0` and `1` remains valid for the sub-one-percent use case but emits one
startup warning explaining that the unit is percentage points and legacy
fraction thresholds need multiplication by 100. Zero, one, and all whole
percentage values emit no warning.

`success_rate` and `cancellation_rate` remain fractions in `[0, 1]`, each over
all terminal window requests. This change does not alter exported metrics,
controller timing, filtering operators, or artifact schema; it changes only
the adaptive evaluator's error-rate value and validation.

## Source anchors

- `rust/runtime/src/adaptive_core/sla.rs` owns metric-family evaluation and
  validation.
- `rust/runtime/src/adaptive_core/controller.rs` owns zero-success evaluation.
- `rust/runtime/src/metrics_core/accumulator.rs` is the authoritative exported
  metric formula.
- `rust/e2e-tests/tests/test_adaptive_scale.rs` exercises the native binary
  against the Rust mock server.
