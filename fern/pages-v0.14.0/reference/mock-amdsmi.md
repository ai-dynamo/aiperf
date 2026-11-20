# Mocking a ROCm Environment

AIPerf collects AMD GPU telemetry through the `amdsmi` collector
(`GPUTelemetryCollectorType.AMDSMI`), a *local* collector that calls the `amdsmi`
Python bindings shipped with ROCm. Those bindings are not pip-installable and
require a real AMD GPU plus driver, so the AMD telemetry path — the `amd_*`
metrics and the console/CSV/plot plumbing — cannot otherwise be exercised on
developer machines, in CI, or on NVIDIA hardware.

The `aiperf-mock-amdsmi` package (under `tests/aiperf_mock_amdsmi/`) installs a
**fake top-level `amdsmi` module** that implements exactly the API surface the
collector consumes and returns static, per-model readings. AIPerf imports it
transparently — nothing in `src/aiperf` changes — so you can benchmark "as if on
an AMD server."

Unlike the NVIDIA DCGM path (an HTTP Prometheus endpoint faked by the mock
server's `/dcgm{N}/metrics`), the AMD collector is local-library-based and cannot
be pointed at an HTTP endpoint. Faking the bindings is therefore the way to drive
the real collector off AMD hardware.

## Install

```bash
make install-mock-amdsmi
# or:
uv pip install -e tests/aiperf_mock_amdsmi
```

It is intentionally not part of the default `make install`.

> **Do not install on real AMD hardware.** Installing this package places a
> top-level `amdsmi` module in your virtualenv that shadows the real ROCm
> bindings at the import level. Even with the dormancy gate active (i.e.
> `AIPERF_MOCK_AMDSMI` unset), `import amdsmi` will raise `OSError` instead of
> loading the real driver bindings, silently suppressing real AMD telemetry.

## Dormancy gate

Even when installed, `import amdsmi` raises `OSError` unless `AIPERF_MOCK_AMDSMI`
is truthy. This mirrors the real "ROCm wheel installed without a working
`libamd_smi.so`" failure that the collector already tolerates
(`except (ImportError, OSError)`), so an installed-but-unactivated fake degrades
to "amdsmi unavailable" rather than silently claiming AMD GPUs exist.

## Configuration

All settings are read from the environment at `amdsmi_init()` time:

| Variable | Default | Meaning |
|---|---|---|
| `AIPERF_MOCK_AMDSMI` | (unset) | Master enable. Must be `1`/`true`/`yes`/`on`. |
| `AIPERF_MOCK_AMDSMI_NUM_GPUS` | `8` | Number of GPU handles to enumerate. |
| `AIPERF_MOCK_AMDSMI_MODEL` | `mi300x` | One of `mi300x`, `mi325x`, `mi355x`, `mi250x`. |
| `AIPERF_MOCK_AMDSMI_GFX_ACTIVITY` | per-model | Override graphics-engine activity (%). |
| `AIPERF_MOCK_AMDSMI_POWER_W` | per-model | Override socket power (W). |
| `AIPERF_MOCK_AMDSMI_TEMP_C` | per-model | Override junction temperature (C). |
| `AIPERF_MOCK_AMDSMI_VRAM_USED_FRACTION` | `0.9` | Fraction of VRAM reported as used. |

Readings are static per model — they do not vary with benchmark traffic. The one
exception is the energy accumulator, which advances monotonically per read so the
cumulative `amd_energy_consumption` delta metric (baseline -> final) is non-zero.

## Example

```bash
# Terminal 1: start the mock inference server
aiperf-mock-server

# Terminal 2: benchmark against it, collecting fake AMD telemetry
AIPERF_MOCK_AMDSMI=1 AIPERF_MOCK_AMDSMI_MODEL=mi300x AIPERF_MOCK_AMDSMI_NUM_GPUS=8 \
  aiperf profile -m <model> --url http://localhost:<port> \
  --gpu-telemetry amdsmi ...
```

The console summary and JSON/CSV exports will show populated `amd_*` metrics
and `platform: "amd"`.
