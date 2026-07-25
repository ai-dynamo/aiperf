<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tutorial: Benchmarking with a generic Hugging Face dataset

AIPerf can benchmark an inference server against **any** Hugging Face dataset,
addressed by its repository ID — you do not need a catalog entry or a code change.
Point AIPerf at the dataset and it downloads the rows, infers which fields hold the
prompt (and, when present, the reference completion), and builds one request per
row.

This tutorial is written against the native `aiperf` binary (crate `aiperf-cli`).
Every flag and default below is taken from the shipped implementation.

---

## 1. Quickstart

Benchmark a chat endpoint against `allenai/WildChat-1M`:

```bash
aiperf profile \
  --hf-dataset allenai/WildChat-1M \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --request-count 200
```

That is the whole thing. `--hf-dataset <id>` is the only new flag required; the rest
(`--model`, `--url`, `--endpoint-type`, `--tokenizer`, request shaping) are the same
flags you already use for synthetic or file datasets.

A tokenizer is **required**: AIPerf tokenizes each prompt locally to compute input
sequence length and to size the output when it is derived from a completion.

---

## 2. What gets auto-detected

When you do not name the columns, AIPerf inspects the **first row** and picks a
layout, applying it to the whole dataset. Detection order:

1. **Chat messages** — a field named `conversation`, `conversations`, or `messages`
   holding an array of chat messages. Both shapes are understood:
   `{"role": "...", "content": "..."}` and the ShareGPT
   `{"from": "...", "value": "..."}` form. The first `user`/`human` message becomes
   the prompt; the first `assistant`/`gpt` message becomes the reference completion.
2. **Joined context** — if a row has both `context` and `input`, they are joined
   (blank-line separated) into one prompt.
3. **Flat prompt** — the first present of, in priority order:
   `prompt`, `question`, `problem`, `input`, `text`, `content`, `instruction`.
   A `turns` array of strings is also accepted (its first entry is used).
4. **Reference completion** (for flat/joined layouts) — the first present of:
   `completion`, `response`, `answer`, `output`, `solution`, `answers`.

If no prompt field can be found, AIPerf fails with a message listing the row's
available fields and suggesting `--hf-text-column`.

> **Detection is first-row-based.** AIPerf assumes the dataset is homogeneous. If a
> dataset mixes shapes row-to-row, detection follows row 0 and rows that do not match
> are skipped. For heterogeneous data, name the field explicitly (Section 4) or split
> the data into homogeneous inputs.

---

## 3. Choosing subset, split, and revision

A Hugging Face dataset can have multiple configs (subsets) and splits.

```bash
aiperf profile \
  --hf-dataset openai/gsm8k \
  --hf-subset main \
  --hf-split test \
  --model ... --url ... --endpoint-type chat --tokenizer ...
```

- `--hf-subset <name>` selects the config/subset.
- `--hf-split <name>` selects the split (`train`, `test`, `validation`, …).
- `--hf-revision <commit|branch|tag>` pins an immutable revision. When set, AIPerf
  resolves the commit and reads the repository's Parquet/JSONL/CSV artifacts rather
  than the live dataset-viewer snapshot.

**If you omit `--hf-subset` / `--hf-split`,** AIPerf resolves them automatically by
querying the dataset viewer's `/info` endpoint: it picks the `default` config (or the
first one), and prefers the split `train`, then `test`, then `validation`, then the
first available. The resolved coordinates are logged.

---

## 4. Overriding the detected columns

Auto-detection is a convenience, not a straitjacket. Override any part:

```bash
aiperf profile \
  --hf-dataset my-org/my-dataset \
  --hf-text-column question_body \
  --hf-output-column reference_answer \
  --model ... --url ... --endpoint-type chat --tokenizer ...
```

- `--hf-text-column <field>` forces the prompt field. If that field holds a message
  array, it is still treated as chat.
- `--hf-output-column <field>` forces the completion field used to size the output.

---

## 5. Controlling the output length

By default the output length per request is derived from the reference completion:
AIPerf tokenizes the completion and uses that token count. If a row has no completion
(and none is configured), it falls back to a default of **128** output tokens.

To pin a fixed output length for every request:

```bash
aiperf profile --hf-dataset <id> --hf-output-len 256 \
  --model ... --url ... --endpoint-type chat --tokenizer ...
```

`--hf-output-len` overrides the completion-derived length.

---

## 6. Filtering rows by token budget

The `hf` loader drops rows that fall outside a token budget. Defaults:

| Filter | Default | Meaning |
|---|---|---|
| `min_sequence_tokens` | 4 | drop prompts shorter than this |
| `max_prompt_tokens` | 1024 | drop prompts longer than this |
| `max_total_tokens` | 2048 | drop rows whose prompt + output exceeds this |

Override any of them with `--dataset-filter key=value` (repeatable). For example, to
allow long-context prompts:

```bash
aiperf profile --hf-dataset <id> \
  --dataset-filter max_prompt_tokens=8192 \
  --dataset-filter max_total_tokens=16384 \
  --model ... --url ... --endpoint-type chat --tokenizer ...
```

`--dataset-filter` is accepted together with `--hf-dataset` (previously it required
`--public-dataset`).

---

## 7. Gated or private datasets

For datasets that require authentication, export a token before running:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx      # or HUGGING_FACE_HUB_TOKEN
aiperf profile --hf-dataset some-org/gated-dataset \
  --model ... --url ... --endpoint-type chat --tokenizer ...
```

AIPerf resolves the token from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the on-disk
Hugging Face token file. A gated/private dataset without a valid token fails with an
`HTTP 401/403` message pointing you at `HF_TOKEN`.

---

## 8. Config-v2 YAML

The same capability is available from a config file. Under `dataset.type: public`,
supply `hf_dataset` instead of a catalog name:

```yaml
dataset:
  type: public
  hf_dataset: openai/gsm8k
  hf_subset: main
  hf_split: test
  hf_output_len: 256          # optional
  hf_text_column: question    # optional
  hf_output_column: answer    # optional
```

Then:

```bash
aiperf profile --config benchmark.yaml
```

`camelCase` aliases (`hfDataset`, `hfSplit`, …) are also accepted.

---

## 9. Forcing a specific loader (advanced)

The auto-detecting `hf` format handles the common shapes. If you need one of AIPerf's
purpose-built public loaders instead — for example the multi-turn conversation loader
or the flat instruction loader — force it with `--hf-format`:

```bash
# Use the conversation loader explicitly, naming its column.
aiperf profile --hf-dataset lmms-lab/LLaVA-OneVision-Data \
  --hf-format hf_conversation \
  --dataset-filter conversation_column=conversations \
  --model ... --url ... --endpoint-type chat --tokenizer ...
```

`--hf-format` accepts any registered loader name (e.g. `hf_conversation`,
`hf_instruction_response`); it defaults to the auto-detecting `hf`.

---

## 10. Trying it on a local file first

The `hf` layout inference is source-agnostic — it reads fields from rows regardless
of where they came from. That means you can validate detection on a **local** JSONL
file, with no network, by selecting the `hf` format on a file input:

```bash
cat > sample.jsonl <<'EOF'
{"prompt": "Explain how a four-stroke engine works.", "completion": "Intake, compression, power, exhaust."}
{"prompt": "What is a tokenizer?", "completion": "It maps text to token ids."}
EOF

aiperf profile \
  --input-file sample.jsonl \
  --custom-dataset-type hf \
  --model ... --url ... --endpoint-type chat --tokenizer ... \
  --num-conversations 2
```

This is a fast way to confirm the field detection and output-length behavior against
your own data before pointing at a large Hub dataset.

---

## 11. Notes and gotchas

- **`--hf-dataset` and `--public-dataset` are mutually exclusive** — use one or the
  other. `--hf-dataset` deliberately bypasses the curated public-dataset catalog.
- **Single-pass vs. recycling.** `--num-conversations N` makes one pass over up to `N`
  rows. `--request-count N` issues `N` requests, recycling the dataset if it has fewer
  than `N` usable rows. Choose based on whether you want each row used once.
- **Row count.** The loader downloads enough rows to satisfy your request/entry count;
  large splits are streamed, not fully loaded.
- **Prompts under 4 tokens are dropped** by default (`min_sequence_tokens`). If a
  dataset of very short prompts yields "no valid samples," lower that filter or pick a
  different field with `--hf-text-column`.
- **Verbatim payloads.** This path composes text prompts (and chat messages) that go
  through normal tokenization and body construction. It is not for datasets of
  pre-serialized raw request bodies — use the `raw_payload` / `inputs_json` inputs for
  those.

---

## See also

- Design record: [`docs/specs/2026-07-25-hf-generic-datasets-design.md`](../../specs/2026-07-25-hf-generic-datasets-design.md)
- Dataset plane: [`docs/specs/dataset.md`](../../specs/dataset.md)
- Front-end ↔ execution boundary: [`docs/specs/runner-protocol.md`](../../specs/runner-protocol.md)
