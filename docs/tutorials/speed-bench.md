---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with SPEED-Bench Dataset
---

# Profile with SPEED-Bench Dataset

AIPerf supports benchmarking using [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) (SPEculative Evaluation Dataset), a benchmark designed for evaluating speculative decoding across diverse semantic domains and input sequence lengths.

This guide covers profiling speculative-decoding-enabled inference servers with SPEED-Bench prompts and assembling a per-category acceptance matrix with `aiperf speed-bench-report`.

---

## Pick your measurement path first

Everything downstream depends on how your server reports acceptance, so decide this before you run anything.

| | Per-request | Server scrape |
|---|---|---|
| **Where the numbers come from** | `metrics.speculative_decoding` on each response body | The server's Prometheus `/metrics` endpoint, sampled during the run |
| **Engines** | vLLM only, direct (not behind Dynamo, which strips the field) | vLLM, SGLang, TensorRT-LLM, NIM-LLM |
| **Runs needed for the 11-category matrix** | **One** | Eleven (one per category) |
| **Scope of the numbers** | Exactly the requests AIPerf issued | The whole server for the sampled window |
| **Extra AIPerf flags** | None | `--server-metrics` (on by default) |
| **Server setup** | `--per-request-spec-decode-metrics summary` | none |

The per-request path is the better measurement wherever it is available: acceptance is attributed to individual requests rather than inferred from a whole-server counter delta, so nothing another client sends during your run can leak into the result, and you get distributions and a pooled histogram instead of a single ratio. It is also what collapses the eleven-run sweep into one run.

If you are on SGLang or TensorRT-LLM, or your vLLM sits behind Dynamo, skip to [Portable path: server scrape](#portable-path-server-scrape). Everything else on this page - the datasets, the prepare step, the sampling defaults - applies to both paths.

---

## Available Dataset Variants

### Aggregate Datasets

These load all categories combined in a single dataset:

| Dataset Name | Samples | Description |
|---|---|---|
| `speed_bench_qualitative` | 880 | All 11 semantic domains combined |
| `speed_bench_throughput_1k` | 1,536 | ~1K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_2k` | 1,536 | ~2K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_8k` | 1,536 | ~8K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_16k` | 1,536 | ~16K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_32k` | 1,536 | ~32K input tokens, all 3 entropy tiers |

On the per-request path these aggregate splits are the ones you want: every request keeps the category it came from, so one run over `speed_bench_qualitative` still produces a per-category breakdown.

### Per-Category Qualitative Datasets (80 prompts each)

Each of the 11 qualitative domains is also registered separately, for running one category in isolation:

| Dataset Name | Category |
|---|---|
| `speed_bench_coding` | Code generation and programming |
| `speed_bench_humanities` | History, philosophy, liberal arts |
| `speed_bench_math` | Mathematical reasoning |
| `speed_bench_multilingual` | Tasks across 23 languages |
| `speed_bench_qa` | Question answering |
| `speed_bench_rag` | Retrieval-augmented generation |
| `speed_bench_reasoning` | Logical and analytical reasoning |
| `speed_bench_roleplay` | Creative roleplay and dialogue |
| `speed_bench_stem` | Science, technology, engineering |
| `speed_bench_summarization` | Text summarization |
| `speed_bench_writing` | Creative and technical writing |

### Per-Entropy-Tier Throughput Datasets (512 prompts each)

Each throughput ISL bucket is also available filtered by entropy tier:

| Pattern | Tiers | Description |
|---|---|---|
| `speed_bench_throughput_{ISL}_low_entropy` | Code, sorting | Predictable output patterns |
| `speed_bench_throughput_{ISL}_mixed` | Needle-in-a-haystack, exams | Moderate unpredictability |
| `speed_bench_throughput_{ISL}_high_entropy` | Creative writing, dialogue | Highly unpredictable output |

Where `{ISL}` is one of: `1k`, `2k`, `8k`, `16k`, `32k`.

---

## Prepare the Dataset

NOTICE: This dataset is governed by the [NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf). For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose. The prepare data script below automatically fetches data from all the source datasets.

You should first download and prepare the dataset using the following one liner:

```bash
SPEED_BENCH_DIR="./datasets/speed-bench"
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py | python3 - --output_dir $SPEED_BENCH_DIR
```

This will download all splits into the working directory as JSONL files. Other supported options of the prepare script:

* `--config`: select which config to prepare, can be one of the splits in the dataset (e.g., `qualitative`, `throughput_2k`) or `all` to prepare all of the configs.
* `--output_dir`: select different output directory to download the dataset to.

> [!IMPORTANT]
> SPEED-Bench stores prompts as references to their upstream source datasets, and the
> prepare script resolves them by downloading each source. **Some of those sources are
> gated on HuggingFace** (for example `cais/hle`), so the script fails partway through
> with `DatasetNotFoundError: Dataset '<name>' is a gated dataset on the Hub` unless you
> have requested access and authenticated:
>
> ```bash
> hf auth login            # or: huggingface-cli login  (older huggingface_hub)
> # or non-interactively:  export HF_TOKEN=hf_...
> ```
>
> Request access on each gated dataset's Hub page, then re-run the prepare script - it
> reuses the HuggingFace cache, so nothing already downloaded is fetched twice. Expect
> several GB of source data and tens of minutes on a first run.
>
> Do not benchmark a partially prepared file: unresolved rows still hold the literal
> placeholder `FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH`.
> AIPerf rejects those rows at dataset load rather than benchmarking the placeholder text,
> but the run fails at startup rather than producing results.

---

## Start a Server with Speculative Decoding

Launch an inference server with speculative decoding enabled. On vLLM, add `--per-request-spec-decode-metrics summary` so each response carries its own acceptance stats:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5, "method": "draft_model"}' \
  --per-request-spec-decode-metrics summary
```

The flag tracks vLLM [PR #48915](https://github.com/vllm-project/vllm/pull/48915); confirm your build includes it. Use `detailed` instead of `summary` to additionally record the ordered per-step accepted/proposed arrays in the records trace. Omit the flag entirely if you are taking the server-scrape path.

Verify the server is ready, and that acceptance metrics are attached (look for `metrics.speculative_decoding`):

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

If that object is absent, the per-request path will silently produce nothing - see [When no acceptance numbers show up](#when-no-acceptance-numbers-show-up).

---

## Recommended Defaults

### Non-Reasoning Models

For standard (non-reasoning) models, use `temperature=0` and a 4K output length cap:

```bash
--osl 4096 --extra-inputs temperature:0
```

Do not set `ignore_eos` - let the model stop naturally at its end-of-sequence token.

### Reasoning Models

For reasoning models (e.g. DeepSeek-R1, QwQ), follow the model card's recommended settings for temperature, top_p, and output length. Reasoning models typically require higher output limits and specific sampling parameters.

---

## Per-Category Matrix in One Run

Run the aggregate qualitative split once. Every request is tagged with the category of the SPEED-Bench row it came from, so the per-request records carry the breakdown with them:

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --custom-dataset-type speed_bench_qualitative \
    --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
    --request-count $(jq -s '[.[].messages | length] | add' ${SPEED_BENCH_DIR}/qualitative.jsonl) \
    --osl 4096 \
    --extra-inputs temperature:0 \
    --concurrency 16 \
    --output-artifact-dir ./artifacts/speed_bench_qualitative
```

> [!WARNING]
> **Size the run by turns, not by prompts.** `--request-count` counts *requests*,
> and a multi-turn SPEED-Bench row dispatches one request per turn - the qualitative
> split's 880 rows expand to roughly 1,140 requests. Passing `--request-count 880`
> therefore stops partway through the file and **silently drops whole categories**,
> which biases the matrix: the categories that fall off the end are not a random
> sample. The `jq` expression above sizes the run to the split's true turn count.
> Without any count at all, AIPerf defaults to 10 requests.
>
> Verify afterwards that every category is present and none was truncated:
>
> ```bash
> jq -r '.metadata.source_kind' ./artifacts/speed_bench_qualitative/profile_export.jsonl \
>   | sort | uniq -c
> ```

No spec-decode flag is needed on the AIPerf side, and `--server-metrics` is unnecessary. The end-of-run console gains a **Spec Decode** section covering the whole run:

> [!IMPORTANT]
> The output below is from a real run of the exact server command shown above
> (Llama-3.1-8B-Instruct target, Llama-3.2-1B-Instruct drafter, `num_speculative_tokens: 5`,
> greedy decoding) over the full qualitative split at concurrency 16. Treat the numbers as
> one data point, not a specification: acceptance depends on the target/drafter pair, the
> draft budget, sampling temperature, the dataset, and the load. Greedy decoding
> (`temperature: 0`) reduces rejection sampling to plain argmax agreement and is the
> acceptance ceiling - the same run at `temperature: 1` will report lower.

```text
                             NVIDIA AIPerf | LLM Metrics: Spec Decode
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃                            Metric ┃    avg ┃   min ┃    max ┃    p99 ┃    p90 ┃   p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│         Acceptance Length (ratio) │   4.19 │  1.83 │   6.00 │   6.00 │   5.35 │  4.13 │   0.81 │
│  Token-Weighted Acceptance Length │   4.63 │   N/A │    N/A │    N/A │    N/A │   N/A │    N/A │
│                           (ratio) │        │       │        │        │        │       │        │
│         Draft Acceptance Rate (%) │  63.83 │ 16.52 │ 100.00 │ 100.00 │  87.03 │ 62.65 │  16.23 │
│ Overall Draft Acceptance Rate (%) │  72.65 │   N/A │    N/A │    N/A │    N/A │   N/A │    N/A │
│     Accepted per Verified (ratio) │   0.70 │  0.30 │   1.00 │   1.00 │   0.89 │  0.69 │   0.14 │
│         Spec Decode Steps (count) │ 118.53 │  1.00 │ 821.00 │ 744.78 │ 230.00 │ 71.00 │ 163.53 │
└───────────────────────────────────┴────────┴───────┴────────┴────────┴────────┴───────┴────────┘
  Accepted drafts per step (% of steps):  0: 13%   1: 9%   2: 7%   3: 6%   4: 4%   5: 61%
```

The accepted-draft histogram is typically **bimodal** rather than smoothly decaying: most verify
steps either accept the whole draft block or almost none of it. That is content-dependent
acceptance - easy stretches where target and drafter agree completely, punctuated by tokens where
they diverge immediately - and it is why the per-request `Acceptance Length` distribution is wide
(`min` 1.83, `max` 6.00 = `num_speculative_tokens + 1`).

Then split that run by category:

```bash
aiperf speed-bench-report ./artifacts/speed_bench_qualitative --format both
```

```text
                                                   Acceptance Length Report
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Model                            ┃ coding ┃ humanities ┃ math ┃ multilingual ┃   qa ┃  rag ┃ reasoning ┃ roleplay ┃ stem ┃ summarization ┃ writing ┃ Overall ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ meta-llama/Llama-3.1-8B-Instruct │   5.01 │       4.84 │ 5.24 │         5.10 │ 4.93 │ 4.51 │      4.47 │     3.34 │ 4.78 │          3.81 │    4.19 │    4.56 │
└──────────────────────────────────┴────────┴────────────┴──────┴──────────────┴──────┴──────┴───────────┴──────────┴──────┴───────────────┴─────────┴─────────┘
CSV written to speed_bench_report.csv
```

The spread is the point of the matrix: acceptance tracks how predictable each domain's output is.
Structured, low-entropy domains (`math`, `multilingual`, `coding`) sit near the top of the range,
while open-ended generation (`roleplay`, `summarization`) sits well below - a 1.9-point gap here
between `math` and `roleplay`. A matrix that comes back flat across all eleven categories is a
signal to check the drafter configuration, not a result.

The acceptance-rate matrix works the same way:

```bash
aiperf speed-bench-report ./artifacts/speed_bench_qualitative --metric accept_rate
```

The same run is available as a config template if you prefer YAML:

```bash
aiperf config init --template speed_bench_per_request --output speed_bench.yaml
aiperf profile --config speed_bench.yaml
```

### How the report finds its numbers

`aiperf speed-bench-report` takes acceptance from the highest-fidelity source each run directory offers, in this order. `--source records|summary|server` pins one explicitly.

| `--source` | File read | Where the data came from | Columns per run |
|---|---|---|---|
| `records` | `profile_export.jsonl` | per-request `metrics.speculative_decoding` on each **response body** | one **per category** |
| `summary` | `profile_export_aiperf.json` | the **same per-request data**, reduced by AIPerf to run-level scalars | one |
| `server` | `server_metrics_export.json` | the engine's **Prometheus counters**, scraped during the run | one |

`records` and `summary` share a provenance - both are the per-request stats AIPerf read from response bodies - and differ only in granularity. `server` is the only source that reads server-side data.

Only `records` can split a single run, because only the per-request trace knows which category each request belonged to. It is written at the default `--export-level records`; a run exported at `--export-level summary` has no trace, so `auto` falls through to `summary` and yields the run-level number as a single column.

Every source computes the same two token-weighted quantities, so columns stay comparable no matter which one produced them:

- `accept_length` - `1 + accepted_draft_tokens / verify_steps`, summed across the category.
- `accept_rate` - `accepted_draft_tokens / proposed_draft_tokens`, summed across the category.

On an engine with no per-request acceptance reporting, `records` and `summary` both come up empty and `auto` falls through to `server` - identical to the behavior before `--source` existed. On vLLM **with** `--per-request-spec-decode-metrics`, `auto` now prefers the per-request numbers, so a matrix regenerated from an older run directory can shift slightly; pass `--source server` to pin the scraped values.

Both weight every verify step equally, matching the console's **Token-Weighted Acceptance Length** and **Overall Draft Acceptance Rate** rather than the per-request means beside them. The two coincide when requests run a similar number of verify steps and diverge when step counts vary and correlate with acceptance. Warmup records are excluded.

Two things to keep in mind when reading the matrix:

- **Cells are turn-weighted, not row-weighted.** Every qualitative category holds 80 rows, but multi-turn rows expand into several requests, so a category's cell can rest on far more requests than another's (in one reference run: `reasoning` 190 requests, `summarization` 80). Later turns also carry accumulated conversation history, which shifts acceptance relative to a first turn.
- **`Overall` is the macro-average of the category cells**, not the pooled run-level value - it weights each category equally, matching how the SPEED-Bench paper reports an overall figure. The console's **Token-Weighted Acceptance Length** is the pooled number over every verify step in the run. Expect them to differ by a few hundredths; a large gap means your categories differ a lot in both acceptance and request volume.

> [!TIP]
> On vLLM you can cross-check the whole thing: `--source server` reduces the run to a single
> column computed from the Prometheus counters. It should match the console's
> **Token-Weighted Acceptance Length** closely - both are pooled over the same requests, by
> two independent routes. A mismatch means either another client was hitting the server
> during your run, or the per-request and scraped paths disagree.


### What a mixed run does and does not change

- **Attribution stays exact.** Each acceptance record is produced by the server for that one request, so interleaving categories cannot mix their numbers.
- **Every category sees the same load.** In an eleven-run sweep each category is measured under a load shaped by its own prompt lengths; in one mixed run they all share one steady state. That is usually the fairer comparison, but it is not identical to the paper's isolated per-category methodology - say which one you ran when you publish numbers.
- **Load-adaptive drafting still applies.** vLLM can vary the draft budget with batch size (`num_speculative_tokens_per_batch_size`) or adapt verification per request. Those effects are driven by the mixed load, not by any one category.
- **Throughput cannot be split.** `--metric throughput` is a rate over the whole run, so it stays one column per run even when the records carry categories. For per-category throughput you still need per-category runs.

### Entropy-tier matrix, also one run

The throughput splits label rows by entropy tier rather than semantic domain, so the same single-run treatment gives you the tier breakdown at a fixed ISL:

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --custom-dataset-type speed_bench_throughput_1k \
    --input-file ${SPEED_BENCH_DIR}/throughput_1k.jsonl \
    --request-count $(jq -s '[.[].messages | length] | add' ${SPEED_BENCH_DIR}/throughput_1k.jsonl) \
    --extra-inputs temperature:0 \
    --concurrency 64 \
    --output-artifact-dir ./artifacts/speed_bench_throughput_1k

aiperf speed-bench-report ./artifacts/speed_bench_throughput_1k
```

The matrix comes back with `low_entropy`, `mixed`, and `high_entropy` columns. Replace `1k` with `2k`, `8k`, `16k`, or `32k` for other input lengths.

---

## Literature Acceptance-Length Datasets (GSM8K, MT-Bench, MATH-500, HumanEval, MBPP)

The speculative-decoding literature overwhelmingly reports acceptance length against five standard benchmarks. AIPerf registers each as a public dataset that is auto-downloaded from HuggingFace at runtime, so there is no prepare-data step: select one with `--public-dataset`.

| Dataset Name | HuggingFace Source | Prompts | Turns | License |
|---|---|---|---|---|
| `spec_al_gsm8k` | `openai/gsm8k` (`main`, `test`) | 1,319 | single | MIT |
| `spec_al_math500` | `HuggingFaceH4/MATH-500` (`test`) | 500 | single | MIT |
| `spec_al_humaneval` | `openai/openai_humaneval` (`test`) | 164 | single | MIT |
| `spec_al_mbpp` | `google-research-datasets/mbpp` (`full`, `test`) | 500 | single | CC-BY-4.0 |
| `spec_al_mtbench` | `HuggingFaceH4/mt_bench_prompts` (`train`) | 80 | two-turn | Apache-2.0 |

These are five separate datasets, so unlike the SPEED-Bench categories they cannot collapse into one run - a run takes one dataset. What the per-request path changes here is only where each run's number comes from: with `--per-request-spec-decode-metrics` enabled, the report reads it from that run's own requests instead of scraping the server, and `--server-metrics` becomes unnecessary.

Prompts are emitted verbatim (the raw question/problem/prompt field); the served model's chat template wraps them at request time via `--endpoint-type chat`. HumanEval and MBPP are text-completion tasks in the spec-decode literature, so chat-wrapping them keeps the matrix uniform but shifts their acceptance length somewhat from the papers' headline numbers. Acceptance length is correctness-agnostic, so use greedy decoding (`--extra-inputs temperature:0`) to match the headline numbers reported in the literature. Note that `--osl` does not apply to public datasets, so cap generation with `--extra-inputs max_tokens:N` instead. `spec_al_mtbench` is multi-turn: AIPerf dispatches both turns per session and feeds the live assistant reply back as conversation history between them - size it with `--num-conversations` rather than `--request-count` (see below).

### Run All Five with a Matrix Report

```bash
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ART=./artifacts/spec-al   # dedicated root so this matrix never merges with speed_bench_* runs

# Single-turn datasets: size each run to the full dataset with --request-count.
for pair in spec_al_gsm8k:1319 spec_al_math500:500 spec_al_humaneval:164 spec_al_mbpp:500; do
  ds="${pair%%:*}"; count="${pair##*:}"
  echo "=== Running dataset: $ds ($count requests) ==="
  aiperf profile \
      --model "$MODEL" \
      --endpoint-type chat \
      --streaming \
      --url localhost:8000 \
      --public-dataset "$ds" \
      --request-count "$count" \
      --extra-inputs temperature:0 max_tokens:4096 \
      --concurrency 16 \
      --output-artifact-dir "$ART/$ds"
done

# MT-Bench is multi-turn (80 two-turn conversations). Size it with
# --num-conversations so every session runs exactly once; --request-count
# recycles the 80 sessions to reach the count and would dispatch each prompt
# more than once.
aiperf profile \
    --model "$MODEL" \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset spec_al_mtbench \
    --num-conversations 80 \
    --extra-inputs temperature:0 max_tokens:4096 \
    --concurrency 16 \
    --output-artifact-dir "$ART/spec_al_mtbench"

# Assemble the acceptance-length matrix (one column per dataset)
aiperf speed-bench-report "$ART" --metric accept_length --format both
```

> Size each run to the full dataset - without an explicit count AIPerf defaults
> to 10 requests. Single-turn datasets use `--request-count`; the multi-turn
> `spec_al_mtbench` uses `--num-conversations 80` (one run per conversation),
> since `--request-count` recycles its 80 sessions to reach the count. Cap
> generation with `--extra-inputs max_tokens:N` (`--osl` is ignored for public
> datasets), and keep these runs in their own artifacts directory so
> `speed-bench-report` does not average them into an unrelated `speed_bench_*`
> matrix. On the server-scrape path, add `--server-metrics http://localhost:8000/metrics`
> to each run.

The report produces one matrix column per dataset:

```text
                         Acceptance Length Report
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Model                      ┃ gsm8k ┃ math500 ┃ mtbench ┃ humaneval ┃ mbpp ┃ Overall ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ meta/llama-3.1-8b-instruct │  2.40 │    2.31 │    1.95 │      2.62 │ 2.55 │    2.37 │
└────────────────────────────┴───────┴─────────┴─────────┴───────────┴──────┴─────────┘
```

The `accept_rate` and `throughput` metrics work identically.

---

## Portable path: server scrape

Use this when the per-request path is unavailable: SGLang, TensorRT-LLM, NIM-LLM containers, or a vLLM behind Dynamo (which strips the custom response field). Acceptance is read from the server's Prometheus endpoint during each run and written to `server_metrics_export.json`; the report computes acceptance length from vLLM's counters (`accepted_tokens / num_drafts + 1`), SGLang's `spec_accept_length` gauge, or a TensorRT-LLM equivalent.

Because the scrape is a whole-server measurement rather than a per-request one, the matrix needs one run per category.

### Server Metrics Endpoint

AIPerf auto-discovers the Prometheus endpoint at `{url}/metrics`. If your server uses a different path, pass it explicitly with `--server-metrics`:

| Server Type | Metrics Path | Flag Needed |
|---|---|---|
| Standalone vLLM / SGLang | `/metrics` (default) | None (auto-discovered) |
| NIM-LLM containers | `/v1/metrics` | `--server-metrics http://localhost:8000/v1/metrics` |

### All 11 Categories

```bash
CATEGORIES="coding humanities math multilingual qa rag reasoning roleplay stem summarization writing"
MODEL="meta/llama-3.1-8b-instruct"

for cat in $CATEGORIES; do
  echo "=== Running category: $cat ==="
  # Size each run to that category's turn count, not its 80 rows.
  turns=$(jq -s --arg c "$cat" \
    '[.[] | select(.category == $c) | .messages | length] | add' \
    ${SPEED_BENCH_DIR}/qualitative.jsonl)
  aiperf profile \
      --model "$MODEL" \
      --endpoint-type chat \
      --streaming \
      --url localhost:8000 \
      --custom-dataset-type speed_bench_${cat} \
      --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
      --server-metrics http://localhost:8000/metrics \
      --request-count "$turns" \
      --osl 4096 \
      --extra-inputs temperature:0 \
      --concurrency 16 \
      --output-artifact-dir "./artifacts/speed_bench_${cat}"
done

aiperf speed-bench-report ./artifacts/ --source server --format both
```

`--source server` is optional but worth passing: it stops the report from silently preferring per-request records if some of those runs happen to have them, so every column in the matrix is measured the same way.

The same eleven runs are also available as a config-driven sweep template, which drives all eleven variations from a single `aiperf profile --config` invocation:

```bash
aiperf config init --template speed_bench_sweep --output speed_bench_sweep.yaml
aiperf profile --config speed_bench_sweep.yaml
```

### Per-entropy-tier scrape

```bash
for tier in low_entropy mixed high_entropy; do
  echo "=== Running throughput_1k tier: $tier ==="
  aiperf profile \
      --model meta/llama-3.1-8b-instruct \
      --endpoint-type chat \
      --streaming \
      --url localhost:8000 \
      --custom-dataset-type "speed_bench_throughput_1k_${tier}" \
      --input-file ${SPEED_BENCH_DIR}/throughput_1k.jsonl \
      --server-metrics http://localhost:8000/metrics \
      --concurrency 64 \
      --benchmark-duration 60
done
```

### Disable Server Metrics

Server metrics collection is enabled by default. On the per-request path it is redundant; turn it off with `--no-server-metrics`:

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --custom-dataset-type speed_bench_qualitative \
    --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
    --no-server-metrics \
    --concurrency 16
```

---

## Throughput per Category

Output-token throughput is a rate over a whole run, so it cannot be attributed to a category inside a mixed run. `--metric throughput` therefore always reports one column per run directory:

```bash
aiperf speed-bench-report ./artifacts/ --metric throughput
```

To compare throughput across categories, run them separately (the eleven-run loop above) and point the report at the parent directory.

---

## When no acceptance numbers show up

An empty matrix, or a run with no **Spec Decode** console section, is expected clean degradation rather than an error. Common causes, per path:

**Per-request path**

- The server was not started with `--per-request-spec-decode-metrics`.
- The vLLM build predates [PR #48915](https://github.com/vllm-project/vllm/pull/48915).
- The server is behind Dynamo, which strips the custom field - use the server-scrape path.
- The run used `n > 1`; vLLM only reports per-request stats for single-sequence requests.
- The run was exported at `--export-level summary`, so there is no `profile_export.jsonl` to split by category. The run-level number is still reported as one column.

**Server-scrape path**

- No `server_metrics_export.json` in the run directory: the run used `--no-server-metrics`, or the endpoint was unreachable.
- The engine exposes acceptance under a metric name the report does not recognize. Inspect `server_metrics_export.json` for a `spec`-flavored metric and file an issue with the name.

**Either path**

- Speculative decoding is off on the server, or the requests produced no verify steps.

---

## See also

- [Per-Request Spec-Decode Metrics](spec-decode-metrics.md) - the per-request path in detail, on any dataset.
- [Speculative Decoding Metrics](../metrics-reference.md#speculative-decoding-metrics) - metric definitions and formulas.
- [Per-Request Speculative-Decoding Acceptance](../reference/spec-decode-acceptance.md) - the engine-neutral record and adapter architecture.
- [SpecBench tutorial](spec-bench.md) - profiling with the SpecBench speculative-decoding dataset.
