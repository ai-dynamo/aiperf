# Agentic Coding Dataset

An agentic coding workload trace that reflects a long-context, KV-reuse-heavy usage pattern across ~1000 multi-turn sessions with a maximum session ISL of ~200k tokens.

## How to Generate

The dataset is generated from the `manifest.json` next to this README, which
ships with the installed package — so the command below works from a repo
checkout or a `pip install aiperf` alike. Synthesis is CPU-only and takes a
couple of seconds.

<!--
`aiperf synthesize` needs no inference server, but the docs-e2e harness has no
serverless group, so this block borrows `tei-rankings`, its cheapest group.
-->

<!-- aiperf-run-tei-rankings-endpoint-server weight=20 -->
```bash
MANIFEST=$(python -c "import aiperf.dataset.agentic_code_gen as m, pathlib; \
  print(pathlib.Path(m.__file__).parent / 'datasets/1k_sessions_200k_ctx/manifest.json')")

aiperf synthesize agentic-code \
  --config "$MANIFEST" \
  --num-sessions 1131 \
  --seed 42 \
  --output ./agentic-dataset
```
<!-- /aiperf-run-tei-rankings-endpoint-server -->

`--output` names a *parent* directory: the run writes into a timestamped
subdirectory of it, such as
`agentic-dataset/manifest_1131s_seed42_20260101-120000/`. That subdirectory
holds `dataset.jsonl` (the trace dataset) and several companion files
documenting the data statistics. Point `--output` somewhere outside the package
directory so a run never writes into the source tree.

## Contents

Included in this directory:

| File | Purpose |
|---|---|
| `manifest.json` | Distribution config + run parameters characterizing the dataset |

Written into the timestamped run directory by the `aiperf synthesize
agentic-code` command above:

| File | Purpose |
|---|---|
| `manifest.json` | Copy of the manifest the run was generated from |
| `dataset.jsonl` | **Mooncake-format trace file** |
| `quality.json` | Per-metric quality stats vs target distribution |
| `report.html` | Full synthesis dashboard |
| `cache_explorer.html` | Interactive prefix-cache structure viewer |
| `simulation.html` | Session-timeline / cache-hit simulation |
| `cache_structure.json` | Raw cache-tree data backing `cache_explorer.html` |
| `comparison.txt` | Text summary comparing target vs realized distributions |
