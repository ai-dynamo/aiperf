# PyNVML Debug Servers

These standalone servers expose the same small HTTP surface for comparing local GPU readings:

- `debug_pynvml_server.py`: calls `pynvml` directly.
- `debug_pynvml_collector_server.py`: samples through AIPerf's `PyNVMLTelemetryCollector`.

Run each server in its own terminal.

```bash
uv run python tools/debug_pynvml_server.py --host 127.0.0.1 --port 8765
```

```bash
uv run python tools/debug_pynvml_collector_server.py --host 127.0.0.1 --port 8766
```

## Capture Native PyNVML Output

```bash
mkdir -p /tmp/aiperf-pynvml-debug/native
BASE=http://127.0.0.1:8765

curl -sS "$BASE/status" \
  | tee /tmp/aiperf-pynvml-debug/native/status-before.json \
  | jq .

curl -sS -X POST "$BASE/start" \
  | tee /tmp/aiperf-pynvml-debug/native/start.json \
  | jq .

curl -sS -X POST "$BASE/sample" \
  | tee /tmp/aiperf-pynvml-debug/native/sample-1.json \
  | jq .

curl -sS -X POST "$BASE/sample" \
  | tee /tmp/aiperf-pynvml-debug/native/sample-2.json \
  | jq .

curl -sS -X POST "$BASE/stop" \
  | tee /tmp/aiperf-pynvml-debug/native/stop.json \
  | jq .
```

## Capture Collector Output

```bash
mkdir -p /tmp/aiperf-pynvml-debug/collector
BASE=http://127.0.0.1:8766

curl -sS "$BASE/status" \
  | tee /tmp/aiperf-pynvml-debug/collector/status-before.json \
  | jq .

curl -sS -X POST "$BASE/start" \
  | tee /tmp/aiperf-pynvml-debug/collector/start.json \
  | jq .

curl -sS -X POST "$BASE/sample" \
  | tee /tmp/aiperf-pynvml-debug/collector/sample-1.json \
  | jq .

curl -sS -X POST "$BASE/sample" \
  | tee /tmp/aiperf-pynvml-debug/collector/sample-2.json \
  | jq .

curl -sS -X POST "$BASE/stop" \
  | tee /tmp/aiperf-pynvml-debug/collector/stop.json \
  | jq .
```

## Compare Key Fields

```bash
jq '{running, gpu_count, total_power_w: .last_sample.total_power_w, total_energy_delta_j: .last_sample.total_energy_delta_j}' \
  /tmp/aiperf-pynvml-debug/native/stop.json \
  /tmp/aiperf-pynvml-debug/collector/stop.json
```

```bash
jq '.samples[] | {gpu_index, gpu_uuid, power_w, total_energy_j, energy_delta_j, gpu_utilization_pct, sm_utilization_pct, memory_used_gb, temperature_c}' \
  /tmp/aiperf-pynvml-debug/native/sample-1.json \
  /tmp/aiperf-pynvml-debug/collector/sample-1.json
```

`energy_delta_j` is measured relative to the baseline captured by `/start`. Use `/stop` to capture one final sample and release NVML.
