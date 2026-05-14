<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Media Mix Benchmarks

Media mix lets you model realistic multimodal workloads where different requests carry different modality combinations (e.g., 60% image, 20% video, 20% audio) and per-modality dimensional variation (e.g., 70% thumbnails + 30% high-res images within the image-request bucket). It also exposes **per-archetype metric breakdowns** in console, JSON, and CSV output so you can see how each request type performs under mixed load.

## Why media mix?

AIPerf's default multimodal benchmarking is all-or-nothing: if `--image-batch-size > 0`, *every* request gets images. That's fine for isolating a single modality but doesn't model production traffic that mixes request types. With media mix you can:

- **Reproduce production traffic patterns** with weighted request archetypes
- **Diagnose contention effects** between modalities (video requests holding large KV-cache allocations can preempt image requests in ways isolated benchmarks miss)
- **Compare serving frameworks** (vLLM vs SGLang) under realistic mixed load instead of single-modality
- **Tune concurrency** with a clear view of which modality bottlenecks first

## Config

Media mix is YAML-only — point at a config file via `--user-config-file`:

```yaml
input:
  media_mix:
    - weight: 0.3
      name: image-and-audio
      text:
        input_tokens: { mean: 100, stddev: 20 }
        output_tokens: { mean: 500 }
      modalities:
        - modality: image
          batch_size: { mean: 2, stddev: 1, min: 1, max: 3 }
          profiles:
            - weight: 0.7
              width: { mean: 1024, stddev: 128 }
              height: { mean: 768, stddev: 96 }
              format: jpeg
            - weight: 0.3
              width: { mean: 256, stddev: 32 }
              height: { mean: 256, stddev: 32 }
              format: png
        - modality: audio
          batch_size: 1
          profiles:
            - weight: 1.0
              length: { mean: 10.0, stddev: 2.0 }
              format: wav

    - weight: 0.5
      name: video-analysis
      text:
        input_tokens: { mean: 2000, stddev: 200 }
        output_tokens: { mean: 1000 }
      modalities:
        - modality: video
          batch_size: 1
          profiles:
            - weight: 1.0
              width: 1280
              height: 720
              duration: 10.0
              fps: 24
              format: mp4

    - weight: 0.2
      name: text-only
      modalities: []
```

### Top-level structure

Each entry in `media_mix` is a **request archetype**. For each request, AIPerf:

1. Samples an archetype by `weight` (probabilities are auto-normalized)
2. For each modality in that archetype, samples a profile by its `weight`
3. Samples `batch_size` from its distribution (see below)
4. Generates the request content using the selected profile properties

### Variable `batch_size`

`batch_size` accepts a fixed integer or a distribution:

```yaml
batch_size: 2                                          # fixed
batch_size: { mean: 3, stddev: 1, min: 1, max: 5 }     # normal, clamped
batch_size: { min: 0, max: 3 }                         # uniform (no mean -> uniform)
```

With `min: 0`, some requests in that archetype get zero items of that modality — useful for modeling "up to N images per request" patterns.

### Per-archetype text overrides

The `text` block is optional:

- Omitted (or `text: true`): text enabled, uses the global `--input-tokens-mean` / `--output-tokens-mean` config
- `text: { input_tokens: { mean: ... }, output_tokens: { mean: ... } }`: text enabled with per-archetype ISL/OSL overrides. Unspecified fields fall back to global.
- `text: false`: text disabled for this archetype (media-only requests)

This lets, say, image-captioning requests carry short prompts (100 tokens) while video-analysis carries long prompts (2000 tokens), within the same benchmark.

### Archetype names

`name` is optional but recommended. Archetypes without a `name` are auto-named `_archetype_0`, `_archetype_1`, etc. by the config validator. Names must be unique within `media_mix` — duplicates fail fast at config load.

## Running

```bash
aiperf profile --user-config-file media_mix_config.yaml --url http://localhost:8000 --model my-model
```

The console prints one Rich table per archetype plus the across-archetype aggregate table:

```
NVIDIA AIPerf | LLM Metrics: image-and-audio (30% of traffic)
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Metric                  ┃   avg ┃   min ┃   max ┃   p95 ┃   p50 ┃
...

NVIDIA AIPerf | LLM Metrics: video-analysis (50% of traffic)
...

NVIDIA AIPerf | LLM Metrics: text-only (20% of traffic)
...

NVIDIA AIPerf | LLM Metrics
(across-archetype aggregate table)
```

## Output files

`profile_export_aiperf.json` (schema 1.2):

- All existing top-level metrics (the across-archetype aggregate) are unchanged
- A new top-level `archetypes` array carries one block per archetype with `archetype_name`, `archetype_weight`, and the same dynamic metric fields the top level uses
- The complete archetype config (profiles, dimensions, formats) is in `input_config.input.media_mix[]` — join by `archetype_name` to enrich the metric blocks

`profile_export_aiperf_archetypes.csv` (new file): tidy/long format with one row per `(archetype, metric, stat)` tuple. Read with `pd.read_csv(path)` and pivot however you need.

```
Archetype,Metric,Unit,Stat,Value
image-and-audio,Request Latency,ms,avg,120.0
image-and-audio,Request Latency,ms,p95,280.0
video-analysis,Request Latency,ms,avg,890.0
...
```

Existing exports (`profile_export_aiperf.csv`, the timeslice files, etc.) are unchanged.

## Interpreting per-archetype results

The aggregate metric is a weighted blend across request types — useful for "did we meet our overall SLA?" but not for diagnosing *which* request type is the problem. Per-archetype metrics give you that diagnostic:

> "Aggregate p95 is 1800ms which violates our 1500ms SLA. Per-archetype: image+audio is 280ms (fine), video-analysis is 2100ms (the violator). Concurrency tuning or admission control for video would fix the aggregate without slowing image requests."

Per-archetype goodput falls out automatically — `GoodputMetric` is `good_request_count / benchmark_duration` and each archetype's bucket has its own good-request count.

## Notes and limits

- **Media mix is synthetic-only.** Custom-file datasets (`--input-file`) don't currently feed archetype names through to records.
- **Keep the archetype count small for readable console output.** &lt;6 archetypes is comfortable; beyond that, the JSON/CSV outputs are still readable but the console gets noisy.
- **Image pool / reuse rate** (sharing images across requests for KV-cache-hit benchmarking) is tracked as a separate feature; it composes with media mix but isn't covered here.
