---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Weights & Biases Export
---

# Weights & Biases Export

## What You Will Learn

This tutorial walks through AIPerf's Weights & Biases (wandb) integration:

- **Post-run results table** — upload the final benchmark results as a
  wandb Table that mirrors the console metrics view, one row per metric
  with `avg`/`min`/`max`/`p99`/`p90`/`p50`/`std` columns.
- **Artifact upload** — attach the generated output files (`inputs.json`,
  `profile_export.jsonl`, CSV/JSON summaries) to the run so it is fully
  reproducible from the wandb UI.
- **Cross-run comparison** — use the run config (model, concurrency,
  request rate) to filter and group benchmark runs in a wandb workspace.

## Prerequisites

Install AIPerf with the optional wandb extra:

```bash
pip install "aiperf[wandb]"
```

Authenticate with wandb if you have not already:

```bash
wandb login   # or export WANDB_API_KEY=...
```

## Run a Profile with wandb Export Enabled

```bash
aiperf profile \
    --url http://localhost:8000 \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --synthetic-input-tokens-mean 128 \
    --output-tokens-mean 128 \
    --concurrency 4 \
    --request-count 64 \
    --wandb-project my-benchmarks \
    --wandb-entity my-team \
    --wandb-run-name baseline-c4 \
    --wandb-tag experiment:baseline
```

### Flag breakdown

| Flag | Effect |
|------|--------|
| `--wandb-project my-benchmarks` | Enables wandb export and selects the project. This is the only required flag. |
| `--wandb-entity my-team` | Team or username owning the project. Defaults to your API key's default entity. |
| `--wandb-run-name baseline-c4` | Run name shown in the wandb UI. Defaults to `aiperf-<benchmark-id>`. |
| `--wandb-tag experiment:baseline` | Additional run tag. Repeat the flag for multiple tags. |

The equivalent config-v2 YAML:

```yaml
benchmark:
  wandb:
    project: my-benchmarks
    entity: my-team
    run_name: baseline-c4
    tags:
      - "experiment:baseline"
```

The exporter runs after profiling completes, once the local exporters
have written their files. The wandb client's local state is written
under the run's artifact directory, not your working directory.

## What Gets Uploaded

### Results table

The run's workspace contains a `summary_metrics` table with the same
metrics, ordering, and labels as the console results table:

| Metric | avg | min | max | p99 | p90 | p50 | std |
|---|---|---|---|---|---|---|---|
| TTFT (ms) | 369.03 | 312.13 | 765.81 | 727.41 | 381.85 | 327.52 | 132.54 |
| Req Latency (ms) | 4,381.82 | 3,936.46 | 4,615.45 | 4,614.24 | 4,603.33 | 4,425.42 | 205.00 |
| Output TPS/User | 249.53 | 232.87 | 277.71 | 276.07 | 261.33 | 246.04 | 12.99 |
| ... | | | | | | | |

Stats a metric does not produce (for example percentiles on count-style
metrics) appear as empty cells, matching the console's `N/A`.

> [!TIP]
> Table panels paginate by default. Use the page-size selector in the
> panel footer (10/25/50/100) to fit the whole table on one page.

### Artifact bundle

Each run logs one `aiperf-run` artifact containing the files from the
run's artifact directory: `inputs.json`, `profile_export.jsonl` (when
record-level export is enabled), `profile_export_aiperf.csv`,
`profile_export_aiperf.json`, plus any parquet, timeslice, or plot
files. Download them from the run's **Artifacts** tab to reproduce or
re-analyze the benchmark.

### Run config and tags

The run config records the endpoint type, model names, redacted URLs,
load generator settings (concurrency, request rate, request count,
duration), and the redacted CLI command. Tags include the AIPerf
version and a `benchmark-<id>` tag. In a project workspace you can
group or filter runs by any config key — for example, group a
concurrency sweep by `phases.0.concurrency`.

## Comparing Runs

Run the same benchmark at several settings, giving each run a
distinguishing name:

```bash
for c in 1 2 4 8; do
  aiperf profile ... \
      --concurrency "$c" \
      --wandb-project my-benchmarks \
      --wandb-run-name "sweep-c${c}"
done
```

In the wandb project, open each run to read its full results table, or
use the runs table to compare the same metric across runs side by side.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `Weights & Biases export is enabled but the optional ... dependency is not installed` | Install the extra: `pip install "aiperf[wandb]"`. |
| `--wandb-entity ... require --wandb-project to be set` | Secondary wandb flags are rejected without `--wandb-project`. |
| Run hangs at exit waiting on wandb | Check network reachability to `api.wandb.ai` and that `WANDB_API_KEY` is valid. |
| No run appears | The exporter only runs when profiling produced results; a benchmark that failed before any request completes skips the upload. |
