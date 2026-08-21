<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native GPU telemetry

## Purpose

Add explicit, local Rust collectors for NVIDIA NVML and AMD SMI without moving
GPU telemetry ownership out of the native runtime. The collectors preserve the
established GPU telemetry record, metric, phase-boundary, and reporting seams;
they replace only the process-local vendor API acquisition that currently lives
behind Python collectors.

## Built

The native runtime already owns GPU telemetry cadence, authoritative profiling
boundary snapshots, accumulation, JSONL output, and the report join. Its public
native collector is DCGM Exporter Prometheus scraping. The lower-level source
seam can also supervise a Python worker, which retains the established Python
`pynvml` and `amdsmi` collectors, but native YAML does not lower either local
collector today.

`DCGM_METRICS` and `AMD_METRICS` define the normalized vendor metric names,
units, scales, and counter-versus-gauge behavior. The accumulator consumes
`GpuScrape` records independently of their source. Consequently, a local
collector must produce the same normalized records and must not introduce a
parallel metrics or exporter path.

## Future requirements

### Explicit collector selection

`gpuTelemetry.collector` gains two native local values:

- `nvml` — NVIDIA Management Library on the process host.
- `amdsmi` — AMD System Management Interface on the process host.

`dcgm` remains the default and keeps its existing HTTP URL behavior. Selection
is explicit: the runtime never falls back from `dcgm` to `nvml`, from `nvml` to
`dcgm`, or between vendors. A native local collector rejects configured DCGM
URLs and a custom DCGM metrics CSV, because neither has a defined local-vendor
meaning. `mode: summary` remains the only native mode.

The protocol-v2 GPU telemetry source union gains `Nvml` and `AmdSmi` variants.
The config projection emits exactly one of those variants for the corresponding
collector name; it emits `Dcgm { url }` only for `dcgm`. The ordinary native
configuration path must construct these variants directly, not route them
through `GpuTelemetrySourceSpec::Python`.

### Source lifecycle

`NvmlTelemetrySource` and `AmdSmiTelemetrySource` implement
`GpuTelemetrySource` in `aiperf_runtime::gpu_telemetry`. Each initializes its
vendor library once while `GpuTelemetryRun::new` constructs the source, retains
stable GPU handles plus metadata, emits one `GpuScrape` per cadence or boundary,
and releases vendor resources from `shutdown`.

The implementation uses runtime-loaded Rust wrappers:

- `nvml-wrapper` for `libnvidia-ml.so.1`.
- `amdsmi` for `libamd_smi.so*`.

Neither library is a link-time requirement of the `aiperf` binary. A missing
library, failed initialization, or no discoverable GPUs marks the selected
source unavailable. The telemetry sidecar logs the structured source error and
continues the benchmark with that source inactive, matching current unavailable
source behavior. It does not select a replacement collector.

Vendor calls remain wholly in the phase sidecar and never in scheduling,
transport, request, token, or metric hot paths. `GpuTelemetrySource::scrape`
uses the sidecar's existing serialized cadence; no per-GPU task, unbounded
channel, or shared request-path lock is introduced.

### Canonical records and metric parity

Both local sources populate `GpuMetadata` with the established platform,
index, UUID, model, and PCI/BDF identity when their vendor APIs expose those
values. A missing optional identity field remains absent; a per-device metadata
failure must not discard metrics obtainable from that device.

NVML writes the existing normalized NVIDIA fields: power in W, cumulative
energy in MJ, GPU and memory utilization in percent, used memory in GB,
temperature in Celsius, encoder and decoder utilization in percent, SM
utilization when supported, and power-violation duration in microseconds. It
also adds `nvidia_jpg_utilization` to the native static field table so the
already-normalized Python-origin field is accumulated natively. Unsupported
NVML functions leave only the affected field absent.

AMD SMI writes the existing normalized AMD fields: power in W, cumulative
energy in MJ, GFX/UMC/MM activity in percent, VRAM used in GB, temperature in
Celsius, uncorrectable ECC count, and throttle status. The collector normalizes
AMD SMI's version- and GPU-dependent power and temperature representations at
the source boundary. Unsupported values and vendor sentinel values remain
absent; they are never coerced to zero.

Energy, ECC, XID, and power-violation fields remain counters and therefore use
only exact opening and closing phase snapshots. All other fields remain cadence
gauges. The existing accumulator alone derives run-level total power, total
energy, output tokens per joule, and energy per user. Per-GPU JSONL and report
schemas do not encode collector-specific payloads or raw vendor structs.

### Failure behavior and observability

Collector selection errors are fail-closed at configuration projection:
unknown collector IDs, local collectors with URLs, and local collectors with a
DCGM custom-metrics file are rejected before execution. After a valid local
source is selected, source initialization and individual vendor API failures
are non-fatal telemetry availability failures. They are emitted with `tracing`
fields identifying the collector and vendor error; request execution continues.

A scrape that yields no records is valid. One failing metric or device must not
suppress a healthy metric or device. A source shutdown error is logged and must
not mask an already-determined benchmark outcome.

### Verification

Unit coverage uses injected vendor API facades, not a local GPU, driver,
network socket, subprocess, or environment-specific library path. It proves:

- config projection and rejection of invalid local-collector combinations;
- one-time initialization, handle metadata, and shutdown;
- conversion and absence semantics for every normalized field;
- exact counter deltas across profiling boundaries and gauge-window summaries;
- partial device and per-metric failures; and
- source-unavailable behavior with no fallback.

Product integration coverage keeps the existing deterministic DCGM exporter
test and adds protocol-level native-source fixture tests that feed the real
sidecar/accumulator path. Hardware validation runs separately on an NVIDIA host
with NVML and an AMD ROCm host with AMD SMI; it verifies records and the
normalized report fields, but is not a unit-test prerequisite.

## Source anchors

- `rust/runtime/src/gpu_telemetry/{source.rs,collector.rs,model.rs,fields.rs,accumulator.rs}`.
- `rust/runtime/src/engine/gpu_telemetry.rs`.
- `rust/runtime/src/engine/{protocol.rs,sidecar_input.rs}`.
- `rust/runtime/src/config/model/telemetry.rs` and `rust/cli/src/{load.rs,yaml.rs,flags.rs}`.
- `rust/e2e-tests/tests/test_gpu_telemetry.rs` and
  `rust/e2e-tests/tests/test_dcgm_faker.rs`.
- `origin/main:src/aiperf/{config/gpu_telemetry.py,gpu_telemetry/pynvml_collector.py,gpu_telemetry/amdsmi_collector.py,gpu_telemetry/constants.py}`.
