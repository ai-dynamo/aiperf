<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Telemetry

## Purpose

Side-channel measurement of the system under test: GPU telemetry, server metrics,
and network latency. These producers feed values into the IO-free metrics seam
(see [metrics.md](metrics.md)); they depend toward it, not the reverse.

## Built

Three modules plus their Clock-paced runner sidecars:

- `aiperf_runtime::gpu_telemetry` — DCGM, native NVML, and native AMD SMI GPU
  sources, exact phase-boundary counters, cadence gauges, and performance/
  accuracy energy/power/efficiency joins. Custom GPU metric bindings are
  supported.
- `aiperf_runtime::server_metrics` — owns a self-contained Prometheus/OpenMetrics
  text parser with Prometheus fallback and auto-disable, a vLLM/SGLang metric
  atlas, and histogram estimation.
- `aiperf_runtime::network_latency` — TCP-connect calibration.

Runtime-owned phase-boundary snapshots replace scrape reconstruction: the sidecars
snapshot at phase boundaries over the run window and are started, driven, and
finished once per cell on the main thread (see
[execution-model.md](execution-model.md)). All cadence and timing routes through
`Clock`.

## Source anchors

- `rust/runtime/src/gpu_telemetry/` (`source.rs`, `nvml.rs`, `amdsmi.rs`,
  `vendor_worker.rs`, `accumulator.rs`, `parser.rs`, `custom_metrics.rs`,
  `fields.rs`).
- `rust/runtime/src/server_metrics/` (`parser.rs`, `prom_text.rs`, `atlas.rs`,
  `histogram.rs`, `accumulator.rs`).
- `rust/runtime/src/network_latency/` (`probe.rs`, `accumulator.rs`).
- `rust/runtime/src/engine/{gpu_telemetry.rs,server_metrics.rs,network_latency.rs}`.
- `rust/e2e-tests/tests/{test_gpu_telemetry.rs,test_server_metrics.rs,test_telemetry_fills.rs,test_network_latency_calibration.rs}`.
