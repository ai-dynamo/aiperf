# Agentic Coding Dataset

An agentic coding workload trace that reflects a long-context, KV-reuse-heavy usage pattern across ~1000 multi-turn sessions with a maximum session ISL of ~200k tokens.

## How to Generate

To generate the dataset, use the directory's `manifest.json` with the following commands:
```bash
aiperf synthesize agentic-code \
  --config manifest.json \
  --num-sessions 1131 \
  --seed 42 \
  --output .
```
This produces a `dataset.jsonl` (the trace dataset) with several companion files documenting the data statistics.

## Contents

Included in this directory:

| File | Purpose |
|---|---|
| `manifest.json` | Distribution config + run parameters characterizing the dataset |

Expected dataset, user-generated via the `aiperf synthesize agentic-code` command above:

| File | Purpose |
|---|---|
| `dataset.jsonl` | **Mooncake-format trace file** |
| `quality.json` | Per-metric quality stats vs target distribution |
| `report.html` | Full synthesis dashboard |
| `cache_explorer.html` | Interactive prefix-cache structure viewer |
| `simulation.html` | Session-timeline / cache-hit simulation |
| `cache_structure.json` | Raw cache-tree data backing `cache_explorer.html` |
| `comparison.txt` | Text summary comparing target vs realized distributions |

