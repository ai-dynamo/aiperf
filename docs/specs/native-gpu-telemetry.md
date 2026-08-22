<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native GPU telemetry

## Purpose

Add explicit, local Rust collectors for NVIDIA NVML and AMD SMI without moving
GPU telemetry ownership out of the native runtime. The collectors preserve the
existing record, metric, phase-boundary, and reporting seams; they replace only
the process-local vendor API acquisition that currently lives behind Python
collectors.

## Built

The native runtime owns GPU telemetry cadence, authoritative profiling boundary
snapshots, accumulation, JSONL output, and the report join. DCGM Exporter
Prometheus scraping, native NVML, and native AMD SMI are the public collector
selections. Native YAML lowers `pynvml` and `amdsmi` to their local native
sources; the native runtime does not launch Python collector workers.

`DCGM_METRICS` and `AMD_METRICS` define the registered normalized metric names,
units, and counter-versus-gauge behavior. Their `scale` values apply only when
the DCGM Prometheus decoder converts exporter values; the accumulator consumes
already-normalized `GpuScrape` records and never applies them. A local source
therefore owns vendor-unit conversion and must not introduce a parallel metrics
or exporter path.

## Future requirements

### Explicit configuration and protocol shape

The stable `gpuTelemetry.collector` vocabulary is `dcgm`, `pynvml`, and
`amdsmi`, matching origin/main. `pynvml` means the native Rust NVML collector;
it does not select or require a Python process. `dcgm` remains the default.
Selection is explicit: the runtime never falls back between DCGM, NVML, and AMD
SMI. Native `pynvml` acceptance is independent of the Python collector's
`nvidia-ml-py` package probe; YAML that must also run through the Python engine
continues to require that Python package there.

`pynvml` and `amdsmi` are local-only YAML selections and reject `urls`. The
existing `--gpu-telemetry <url-or-csv>` surface remains DCGM-only in this slice;
it does not select a collector. Native YAML must additionally plumb its
otherwise-unset `metrics_file` field so that a local selection can reject a
custom DCGM CSV before execution. A later dedicated collector CLI flag is out
of scope.

The lowered `GpuSource` is reshaped into a tagged enum (or equivalent
URL-optional representation) so `dcgm` alone serializes a required URL and
local selections serialize no URL. The strict protocol-v2
`GpuTelemetrySourceSpec` in `engine/sidecar_input.rs` gains URL-less `Nvml` and
`AmdSmi` variants. The config projection maps the three collector IDs directly
to `Dcgm { url }`, `Nvml`, or `AmdSmi`; it must not route through a Python
telemetry worker.

### Source lifecycle and isolation

`NvmlTelemetrySource` and `AmdSmiTelemetrySource` implement
`GpuTelemetrySource` in `aiperf_runtime::gpu_telemetry`. Each retains immutable
device identity (index plus metadata), emits one `GpuScrape` per cadence or
boundary, and releases the library from `shutdown`.

Vendor FFI is not run on the phase runner's current-thread `LocalSet`. Each
source owns a dedicated blocking worker thread that initializes its library,
performs all enumeration and FFI calls, and shuts it down. The `!Send` source
forwards bounded scrape/shutdown requests and awaits Clock-bounded replies; it
never calls synchronous vendor FFI itself. A timeout abandons only the caller
wait: the worker remains owned by the global reaper until it exits, because a
vendor FFI call cannot be interrupted safely. This keeps the co-located
`workers == 1` request issuer responsive while preserving Rust-owned `Clock`
cadence and phase barriers. Thread startup, request, response, timeout, and
shutdown failures surface as `GpuTelemetryError`.

NVML uses `nvml-wrapper` for its dynamically loaded base API and a narrow,
versioned raw `nvml-wrapper-sys` calls where the safe wrapper lacks required
functions (JPEG utilization and GPM SM utilization). NVML devices are
re-resolved by retained index inside the worker for each scrape; no
self-referential `Device<'nvml>` is stored. GPM uses worker-owned paired opaque
samples, is preferred for SM utilization, and rotates samples after every
attempted read. If GPM is unavailable, the process-utilization fallback retains
a per-device timestamp and requests only newer driver-buffered samples.

AMD SMI uses direct, dynamically loaded `libamd_smi.so*` FFI generated and
verified from the target ROCm installation's AMD SMI headers before a dependency
is selected. The current high-level `amdsmi` 0.1.0 crate is not sufficient: it
lacks temperature, ECC, throttle, BDF, and the raw power-information variants
needed here. The implementation may use a generated binding only after proving
it exposes the complete required API; otherwise it owns the narrow required FFI
surface. Neither selected vendor library is a link-time requirement of `aiperf`.

A missing library, initialization failure, or no devices makes the selected
source unavailable. `GpuTelemetryRun::new` logs the construction failure with
`collector` and `error`; a source that later fails its opening or closing
boundary scrape is logged by the sidecar with `source` and `error`. Neither
case selects a replacement collector or changes benchmark execution.

### Canonical records, identity, and metrics

Local NVML records use `pynvml://localhost`; AMD SMI records use
`amdsmi://localhost`, matching origin/main identifiers. These values remain the
`GpuSeriesKey` source identity. The JSONL schema continues to write that value
under its historical `dcgm_url` key for compatibility; the key name does not
imply that every source is DCGM.

Both sources populate platform, index, UUID, model, and PCI/BDF identity when
available. Missing optional identity fields remain absent; a metadata failure
must not discard independently obtainable metrics.

NVML emits already-normalized values: mW to W (`1e-3`), mJ to MJ (`1e-9`),
bytes to GB (`1e-9`), GPU/memory/encoder/decoder/JPEG utilization in percent,
temperature in Celsius, SM utilization in percent, and violation nanoseconds
to microseconds (`1e-3`). It emits the established NVIDIA power, energy, GPU
and memory utilization, used memory, temperature, encoder, decoder, SM, JPEG,
and power-violation fields. `nvidia_jpg_utilization` is added to the static
registered field table. Unsupported functions leave only that field absent.
XID remains DCGM-only.

AMD SMI emits already-normalized values: power in W; energy accumulator times
its resolution in microjoules to MJ (`1e-12`); activity in percent; VRAM bytes
to GB (`1e-9`); and temperature in Celsius. It uses the raw power-information
fields in precedence order `socket_power`, `current_socket_power`, then
`average_socket_power`; accepts AMD SMI's documented version/GPU temperature
representations at the source boundary; and emits GFX/UMC/MM activity,
uncorrectable ECC, and throttle status where supported. Vendor sentinel values
and unsupported fields remain absent, never zero.

Energy, ECC, power violation, and DCGM XID are counters. Every boundary scrape
must return `Ok(Some(GpuScrape))`, including an empty-record scrape, because
boundary collection constructs exact counter snapshots. A violation returns a
`GpuTelemetryError`; production boundary collection must not retain an
`expect()` for this condition. `Ok(None)` is reserved only for duplicate
continuous DCGM bodies. All other fields are cadence gauges. `mode: summary` is
the only native display mode. The existing accumulator alone derives total
power, total energy, output tokens per joule, and energy per user.

### Validation and verification

Unknown collector IDs, local collectors with URLs, local collectors with a
custom DCGM metrics file, malformed lowered local source variants, and an
invalid mode fail closed before execution. A valid source's construction,
per-device, per-metric, cadence, boundary, or shutdown failure is a telemetry
availability failure: log it structurally and continue request execution. One
failing metric or device must not suppress healthy devices or metrics.

Unit tests inject a vendor-worker facade; they do not require a GPU, driver,
socket, subprocess, or host library. Coverage proves configuration projection;
local-source serialization; worker-thread lifecycle; identity; per-field
normalization and absence semantics; boundary `Some(empty)` behavior; counter
deltas; gauge summaries; partial failures; JPEG raw-FFI behavior; and no
fallback. Product tests feed fixture sources through the real sidecar and
accumulator. Unit and fixture tests require no vendor hardware. Because this
development environment has NVIDIA hardware, NVML hardware validation is a
required acceptance check before merge and verifies normalized JSONL and report
fields. AMD SMI hardware validation remains an opt-in external acceptance check
when an AMD ROCm host is available.

## Source anchors

- `rust/runtime/src/gpu_telemetry/{source.rs,collector.rs,model.rs,fields.rs,accumulator.rs,nvml.rs,amdsmi.rs,vendor_worker.rs}`.
- `rust/runtime/src/engine/{gpu_telemetry.rs,sidecar_input.rs}`.
- `rust/runtime/src/config/model/telemetry.rs` and `rust/cli/src/yaml.rs`.
- `rust/e2e-tests/tests/{test_gpu_telemetry.rs,test_dcgm_faker.rs}`.
- `origin/main:src/aiperf/{config/gpu_telemetry.py,plugin/plugins.yaml,gpu_telemetry/pynvml_collector.py,gpu_telemetry/amdsmi_collector.py,gpu_telemetry/constants.py}`.
