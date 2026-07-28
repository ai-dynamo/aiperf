---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Per-Request Spec-Decode Metrics
---

# Per-Request Speculative-Decoding Metrics

When an inference server reports **per-request** speculative-decoding acceptance on
each response, AIPerf turns it into an end-of-run **Spec Decode** console section, the
same scalars in the aggregate JSON/CSV, a pooled acceptance histogram in JSON, and
per-request values in the records trace. It is **engine-agnostic** -- AIPerf reads a
normalized acceptance record and never branches on the engine -- and needs **no AIPerf
flag**: the section appears automatically whenever the stats are present, and disappears
cleanly when they are not.

> [!NOTE]
> There are two ways to get acceptance metrics out of AIPerf; pick by what your server exposes:
>
> - **Per-request (this guide):** the stats ride the response body, per choice, so AIPerf
>   reads them inline during a normal `aiperf profile` run. Highest fidelity -- you get
>   per-request distributions (avg/min/percentiles) and a pooled histogram. Direct-to-vLLM only.
> - **Server scrape:** aggregate acceptance is read from the server's Prometheus `/metrics`
>   endpoint via `--server-metrics` and assembled with `aiperf speed-bench-report`. See the
>   [SPEED-Bench tutorial](speed-bench.md) for that path and for ready-made datasets.

For the record shape and the adapter that fills it, see
[Per-Request Speculative-Decoding Acceptance](../reference/spec-decode-acceptance.md); for
the metric definitions and formulas, see the
[Speculative Decoding Metrics](../metrics-reference.md#speculative-decoding-metrics) section
of the metrics reference.

---

## Prerequisites

- A vLLM server running **speculative decoding** with **per-request stats enabled** via
  `--per-request-spec-decode-stats summary` (or `detailed`). The field shape tracks vLLM
  [PR #48915](https://github.com/vllm-project/vllm/pull/48915) -- confirm your vLLM build
  includes it.
- **Direct-to-vLLM only.** Behind Dynamo the custom stats field is currently stripped, so
  the per-request path is unavailable there (use the server-scrape path instead).
- Streaming works out of the box; per-request `completion_tokens` is only populated when
  the server also returns usage (enable server token counting if you want it).

---

## Start a vLLM server with per-request spec-decode stats

This example uses a Llama-3.1-8B target with a Llama-3.2-1B draft model and a 5-token draft
budget. `--per-request-spec-decode-stats` is the flag that makes vLLM attach acceptance
stats to each response choice:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5, "method": "draft_model"}' \
  --per-request-spec-decode-stats summary
```

Verify the server is ready:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Run AIPerf

No spec-decode-specific flag is required -- run a normal profile. Add `--export-level
records` if you want the per-request acceptance struct in the records trace (see
[Per-request trace](#per-request-trace) below):

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 512 \
    --output-tokens-mean 256 \
    --concurrency 16 \
    --export-level records
```

---

## Reading the output

### Console: the Spec Decode section

The end-of-run console gains a dedicated **Spec Decode** table followed by a one-line
pooled acceptance histogram.

> [!IMPORTANT]
> The table below is **illustrative -- it shows the output format, not a benchmark
> result.** The numbers are placeholders; real values depend entirely on your model,
> drafter, draft budget, dataset, and concurrency.

```text
                                    NVIDIA AIPerf: Spec Decode
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃                            Metric ┃    avg ┃   min ┃    max ┃    p99 ┃    p90 ┃    p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│         Acceptance Length (ratio) │   3.21 │  1.00 │   6.00 │   5.80 │   4.90 │   3.10 │  1.05 │
│  Token-Weighted Acceptance Length │   3.20 │   N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
│                           (ratio) │        │       │        │        │        │        │       │
│         Draft Acceptance Rate (%) │  44.20 │  0.00 │ 100.00 │  96.00 │  82.00 │  42.00 │ 18.00 │
│ Overall Draft Acceptance Rate (%) │  46.80 │   N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
│     Accepted per Verified (ratio) │   0.53 │  0.17 │   1.00 │   0.97 │   0.82 │   0.52 │  0.17 │
│         Spec Decode Steps (count) │ 128.00 │ 40.00 │ 410.00 │ 390.00 │ 250.00 │ 118.00 │ 60.00 │
└───────────────────────────────────┴────────┴───────┴────────┴────────┴────────┴────────┴───────┘
  Accepted drafts per step (% of steps):  0: 25%   1: 18%   2: 15%   3: 13%   4: 12%   5: 17%
```

Quick reading (full definitions in the
[metrics reference](../metrics-reference.md#speculative-decoding-metrics)):

- **Acceptance Length** -- tokens emitted per verify step (`j + 1`); the headline speed
  number. **Token-Weighted Acceptance Length** is the run-level companion that weights every
  verify step equally instead of every request.
- **Draft Acceptance Rate** -- fraction of proposed draft tokens accepted (`j / l`); the
  drafter-quality number. **Overall Draft Acceptance Rate** is its draft-volume-weighted
  run-level companion.
- **Accepted per Verified** -- `(j + 1) / (l + 1)`, a `[~0, 1]` utilization: how close each
  step got to accepting everything it proposed.
- **Spec Decode Steps** -- verify steps per request.
- **Accepted-draft histogram** -- share of verify steps that accepted exactly `j` draft
  tokens, pooled across the run. Capped to buckets `0..7` on the console (any `j >= 8` folds
  into a trailing `>=8` bucket); the full histogram is in the JSON export.

The section is omitted entirely when no request carried spec-decode stats.

### Aggregate JSON and CSV

The scalar metrics land in `profile_export_aiperf.json` and `profile_export.csv`. The full
pooled histogram is structured, so it goes to JSON only, under
`pooled_spec_decode_acceptance_histogram` (its counts sum to `total_spec_decode_steps`):

> [!NOTE]
> Illustrative values.

```json
"pooled_spec_decode_acceptance_histogram": {
  "0": 6400, "1": 4608, "2": 3840, "3": 3328, "4": 3072, "5": 4352
}
```

### Per-request trace

At `--export-level records`, each line of `profile_export.jsonl` carries the neutral
acceptance struct under `spec_decode_acceptance`, so the per-request histogram, counts, and (when the
server reports them) per-step arrays travel with the trace:

> [!NOTE]
> Illustrative values.

```json
{
  "metadata": { "...": "..." },
  "metrics": { "...": "..." },
  "spec_decode_acceptance": {
    "engine": "vllm",
    "mean_acceptance_length": 3.18,
    "draft_acceptance_rate": 0.436,
    "acceptance_histogram": {"0": 34, "1": 22, "2": 19, "3": 16, "4": 15, "5": 22},
    "num_accepted_draft_tokens": 281,
    "num_draft_tokens": 640,
    "num_spec_steps": 128,
    "num_spec_tokens": 5
  }
}
```

At `--export-level raw`, the histogram is already present in the raw response body that raw
export preserves, so no additional field is needed there.

---

## When nothing shows up

If the Spec Decode section, histogram, and `spec_decode_*` fields are all absent, that is
the expected clean-degradation behavior -- not an error. Common causes:

- speculative decoding is off, or the requests had no verify steps;
- the server was not started with `--per-request-spec-decode-stats`;
- the server is behind Dynamo, which strips the custom field (use the
  [server-scrape path](speed-bench.md) instead);
- the vLLM build predates [PR #48915](https://github.com/vllm-project/vllm/pull/48915).

---

## See also

- [Speculative Decoding Metrics](../metrics-reference.md#speculative-decoding-metrics) -- metric definitions and formulas.
- [Per-Request Speculative-Decoding Acceptance](../reference/spec-decode-acceptance.md) -- the engine-neutral record and adapter architecture.
- [SPEED-Bench tutorial](speed-bench.md) -- the server-scrape acceptance path and ready-made speculative-decoding datasets.
- [SpecBench tutorial](spec-bench.md) -- profiling with the SpecBench speculative-decoding dataset.
