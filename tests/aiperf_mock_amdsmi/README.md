# aiperf-mock-amdsmi

Fake `amdsmi` bindings that let AIPerf's AMD ROCm telemetry path run on hardware
that has no AMD GPU (developer laptops, CI, NVIDIA boxes).

## Why

AIPerf collects AMD GPU telemetry through `AMDSMITelemetryCollector`, a *local*
collector that calls the `amdsmi` Python bindings shipped with ROCm. Those
bindings are not pip-installable and require a real AMD GPU plus driver, so the
entire AMD code path (the `amd_*` metrics, `platform="amd"`, and the
console/CSV/plot plumbing) is otherwise impossible to exercise off AMD hardware.

This package installs a top-level `amdsmi` module that implements exactly the API
surface the collector consumes and returns static, per-model readings. AIPerf
imports it transparently — **nothing in `src/aiperf` changes**.

## Install

```bash
make install-mock-amdsmi
# or directly:
uv pip install -e tests/aiperf_mock_amdsmi
```

It is deliberately **not** part of the default `make install`, so `amdsmi` only
appears when you opt in.

## Dormancy gate

Even when installed, `import amdsmi` raises `OSError` unless `AIPERF_MOCK_AMDSMI`
is truthy. This mirrors the real "ROCm wheel installed without a working
`libamd_smi.so`" failure that the collector already tolerates, so an
installed-but-unactivated fake degrades to "amdsmi unavailable" rather than
silently claiming AMD GPUs exist.

## Configuration

All configuration is read from the environment at `amdsmi_init()` time:

| Variable | Default | Meaning |
|---|---|---|
| `AIPERF_MOCK_AMDSMI` | (unset) | Master enable. Must be `1`/`true`/`yes`/`on` to activate. |
| `AIPERF_MOCK_AMDSMI_NUM_GPUS` | `8` | Number of GPU handles to enumerate. |
| `AIPERF_MOCK_AMDSMI_MODEL` | `mi300x` | One of `mi300x`, `mi325x`, `mi355x`, `mi250x`. |
| `AIPERF_MOCK_AMDSMI_GFX_ACTIVITY` | per-model | Override graphics-engine activity (%). |
| `AIPERF_MOCK_AMDSMI_POWER_W` | per-model | Override socket power (W). |
| `AIPERF_MOCK_AMDSMI_TEMP_C` | per-model | Override junction temperature (C). |
| `AIPERF_MOCK_AMDSMI_VRAM_USED_FRACTION` | `0.9` | Fraction of VRAM reported as used. |

Readings are static per model. The one exception is the energy accumulator,
which advances monotonically per read so the cumulative `amd_energy_consumption`
delta metric (baseline -> final) is non-zero.

## Example: benchmark "as if on an AMD server"

```bash
# Terminal 1: start the mock inference server
aiperf-mock-server

# Terminal 2: benchmark against it, collecting fake AMD telemetry
AIPERF_MOCK_AMDSMI=1 AIPERF_MOCK_AMDSMI_MODEL=mi300x AIPERF_MOCK_AMDSMI_NUM_GPUS=8 \
  aiperf profile -m <model> --url http://localhost:<port> \
  --gpu-telemetry amdsmi ...
```

The console summary and JSON/CSV exports will show populated `amd_*` metrics.
Once the GPU-telemetry vendor-namespacing work lands, `platform: "amd"` will also
appear.
