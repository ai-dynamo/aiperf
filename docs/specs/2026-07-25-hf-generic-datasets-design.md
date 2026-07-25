<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Generic HuggingFace datasets by ID (design)

## Purpose

Let a user point AIPerf at an arbitrary HuggingFace dataset by repository ID —
`--hf-dataset allenai/WildChat` — and have the prompt (and, when present, the
completion) columns auto-detected, with no catalog entry and no code change. This
closes the one dataset-input UX gap against vLLM's Rust benchmark tool
(`--dataset-name hf --dataset-path <id>`), while keeping AIPerf's already-superior
HF transport (parquet/JSONL/CSV shard streaming, revision pinning, token-file auth)
underneath.

This is a design record for **unbuilt** work. The current state and the exact code
seams it builds on are in [dataset.md](dataset.md) and
[runner-protocol.md](runner-protocol.md).

## Built (today, for context)

Arbitrary-ID HF download and generic column-mapped composition already exist in the
runtime; only the CLI/Config front door is missing:

- `DatasetSource::HuggingFace { dataset, config, split, max_rows, revision }`
  (`rust/runtime/src/dataset/loader/mod.rs:70`) already carries every selection
  option — dataset ID, subset/config, split, revision, row cap.
- The fetch path (`load_hugging_face_rows`, `rust/runtime/src/dataset/loader/public.rs`)
  hits the Dataset Viewer `/rows` REST API and falls back to `hf-hub` parquet/JSONL/CSV
  shard streaming for revision-pinned loads; HF-token resolution (`resolve_hf_token`,
  `mod.rs:130`) reads `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env and the on-disk token file.
- Generic **configured** composers exist: `hf_instruction_response`
  (`HfInstructionComposer`, needs an explicit `prompt_column`) and `hf_conversation`
  (`HfConversationComposer`, needs an explicit `conversation_column`).
- The runtime lowering (`build_public_dataset`,
  `rust/runtime/src/engine/execute/dataset_build.rs:516`) constructs a dataset from a
  `Dataset::Public { name, format, source, options }` and **never consults the CLI
  catalog** — it only requires `name` non-empty, `format` a registered loader, and a
  `source` object satisfying `PublicDatasetSourceSpec` (`deny_unknown_fields`, HF
  shape `{type:hugging_face, dataset, subset, split, revision?}`).

The "hard-coded catalog" wall is therefore entirely CLI-side, in three spots:
`load.rs:931` (`public_catalog::lookup`), `yaml.rs:1277` (requires a `dataset:`
catalog name), and — because HF sources return an **empty probe** with no fetch
(`loader/mod.rs:520`) — the `detect`/`probe` seam cannot content-sniff a HF source,
so a column-auto-detecting loader must be reached by an explicit `format` and do its
own inference inside compose.

## Future requirements (this design)

### Scope

- **In:** arbitrary-ID passthrough **and** column auto-detection, on the Rust CLI
  and the Config-v2 YAML surface. Native-only — no Python/Pydantic change (the
  native `--public-dataset`/dataset build path never round-trips through Python).
- **Out (v1):** multimodal (image/video/audio) auto-detection — those stay reachable
  via `--hf-format hf_conversation` / catalog entries; no new caching layer; no
  Python Config-v2 schema change.

### Component 1 — `HfAutoComposer` (new format id `hf`)

A new `DatasetFormatRegistration` registered in `register_builtin_formats`
(`rust/runtime/src/dataset/loader/mod.rs:365`). The composer is **source-agnostic**:
it turns already-fetched `RawRow`s into requests by inspecting their keys, so the
same logic runs over HF-fetched rows *and* a local JSONL fixture (the property that
makes it offline-testable — see Testing).

**Detection** (on the first row, priority order, a port of vLLM's
`detect_column_format`):

1. **Chat** — a column named `conversation` / `conversations` / `messages` whose
   value is an array of chat messages (`{role, content}` or ShareGPT `{from, value}`).
   Emits a chat/messages request; first `user`/`human` message is the prompt, first
   `assistant`/`gpt` message is the completion.
2. **Combined** — both `context` and `input` present → prompt is `context` + `input`
   joined with `\n\n`.
3. **Text** — first present of `prompt, question, problem, input, text, content,
   instruction`; completion from first present of `completion, response, answer,
   output, solution, answers`. Special-cases a `turns[]` array (take first) and an
   `answers[]` array (take first).
4. **No match** → error listing the row's available columns and suggesting
   `--hf-text-column` (reuses the existing helpful-error idiom at `public.rs:393`).

**Options read** (via the existing `string_option`/`bool_option`/`usize_option`
helpers; every one settable from a CLI flag or `--dataset-filter key=value`):

| Option key | Meaning | Default |
|---|---|---|
| `text_column` / `prompt_column` | force the prompt column | auto-detect |
| `output_column` | force the completion column | auto-detect |
| `conversation_column` | force the chat column | auto-detect |
| `output_len` | fixed output length (overrides derivation) | derive from completion, else 128 (warn once) |
| `min_sequence_tokens` | drop prompts shorter than this | 4 |
| `max_prompt_tokens` | drop prompts longer than this | 1024 |
| `max_total_tokens` | drop prompt+completion over this | 2048 |

Output-length rule (vLLM parity): `output_len` if set; else tokenize the detected
completion and use its length; else default 128 with a single warning.

### Component 2 — arbitrary-ID HF fetch + split/config resolution

Reuses `load_hugging_face_rows` unchanged for the download. Adds one piece:
**when subset/split are omitted**, resolve them via the Dataset Viewer `/info`
endpoint before fetching (default config = `default` or first; split priority
`train > test > validation > first`) — matching vLLM. Because detection needs no
fetch of its own, this resolution runs inside the loader path for a
`DatasetSource::HuggingFace` whose `config`/`split` are unset, using the existing
reqwest client + HF-token headers.

All selection options are already representable on `DatasetSource::HuggingFace`, so
"support all the options (train/test/validation/…, subset, revision, row cap)" is a
matter of surfacing them as flags, not new runtime plumbing.

### Component 3 — CLI and Config-v2 YAML wiring

**New CLI flags** (`rust/cli/src/flags.rs`), projected into `Inputs`
(`rust/cli/src/load.rs`):

| Flag | Maps to |
|---|---|
| `--hf-dataset <id>` | `source.dataset` |
| `--hf-subset <cfg>` (exists) | `source.subset` (→ runtime `config`) |
| `--hf-split <split>` | `source.split` (else auto-resolved) |
| `--hf-revision <rev>` | `source.revision` |
| `--hf-text-column <col>` | option `text_column` |
| `--hf-output-column <col>` | option `output_column` |
| `--hf-output-len <n>` | option `output_len` |
| `--hf-format <fmt>` | escape hatch: force `hf_conversation`/`hf_instruction_response` instead of auto `hf` |

Row cap (`max_rows`) continues to come from the existing `--entries` /
`--request-count` path (`max_conversations`), unchanged.

**`load.rs` branch:** a new arm at the top of the dataset-kind selection chain
(before the `load.rs:931` public branch). When `inputs.hf_dataset` is set, build
`Dataset::Public` directly — `name = <id>`, `format = "hf"` (or `--hf-format`),
`source = {type:hugging_face, dataset:<id>, subset:<--hf-subset or "default">,
split:<--hf-split or "">, revision:<--hf-revision>}`, `options` from the column/output
flags merged with any `--dataset-filter` — **skipping `public_catalog::lookup`
entirely**. Relax `parse_dataset_filters` so `--dataset-filter` is accepted with
`--hf-dataset` too (not only `--public-dataset`).

**YAML (`rust/cli/src/yaml.rs`):** add `hf_dataset` / `hf_split` / `hf_revision` /
`format` / `options` to `DatasetSection` under `dataset.type: public`, and relax the
`yaml.rs:1277` `ensure!` from "requires a `dataset:` catalog name" to "requires a
`dataset:` catalog name **or** `hf_dataset`". Both surfaces funnel into the same
`Inputs` HF path.

### Error handling

- Unknown / gated / private dataset: existing 401/403 → "may be gated or private; set
  HF_TOKEN" messaging.
- No detectable column: error listing available columns, suggest `--hf-text-column`.
- Empty result after token-length filtering: explicit error.
- `--hf-dataset` and `--public-dataset` both set: reject (mutually exclusive).
- `--dataset-filter` without `--hf-dataset` or `--public-dataset`: existing rejection.

### Testing

Per the project's per-record E2E requirement, but network-free:

- **Unit tests** for `detect_column_format` + `HfAutoComposer` over fixture
  `RawRow`s — port vLLM's ~30 cases (chat role/content, ShareGPT from/value, combined
  context+input, turns[], answers[], user override, missing column, output-len
  derivation, short-prompt filtering, oversample, disable-shuffle).
- **Offline E2E** against the in-repo `aiperf-mock-server`: because the composer is
  source-agnostic, a local JSONL fixture selected with `format:"hf"` drives the full
  compose → request → per-record path; assert ISL/OSL/model/streaming/prompt-content
  per record against a deterministic mock config.
- **`#[ignore]`d network test** for live `/info` split/config resolution + `/rows`
  fetch against a small public dataset.

## Prior art — vLLM Rust bench comparison

vLLM's `rust/src/bench/src/datasets/hf_dataset.rs` (`--dataset-name hf
--dataset-path <id>`) is the reference UX this design reaches parity with.

| Capability | vLLM Rust bench | This design |
|---|---|---|
| Arbitrary HF ID, no catalog/code change | ✓ | ✓ (`--hf-dataset`) |
| Auto config/split resolution (`/info`) | ✓ (`train>test>validation>first`) | ✓ (same) |
| Column auto-detection | `detect_column_format` | `HfAutoComposer` (port) |
| Chat detect (role/content + from/value) | ✓ | ✓ |
| Combined `context`+`input` | ✓ | ✓ |
| Text/output column priority lists | ✓ | same lists |
| Completion→output-len, else default 128 | ✓ | ✓ |
| Explicit column override | `--hf-text-column` | `--hf-text-column` / `--dataset-filter` |
| Fixed output length | `--hf-output-len` | `--hf-output-len` |
| Short-prompt filter (<4 tokens) | ✓ | `min_sequence_tokens=4` |
| HF_TOKEN + gated 401/403 messaging | ✓ | ✓ (existing) |
| Shuffle + oversample | ✓ | ✓ (existing samplers) |
| **Parquet/JSONL/CSV shard streaming** | ✗ (REST `/rows` only) | ✓ (existing, exceeds vLLM) |
| **Revision pinning** | ✗ | ✓ (existing) |
| **On-disk token-file auth** | env only | ✓ (existing) |
| **Source-agnostic auto-detect (local JSONL)** | ✗ (welded to download) | ✓ |
| `--hf-output-column` override | ✗ | ✓ |
| Multimodal auto-detection | ✗ | ✗ (v1 out of scope) |

Net: parity on the arbitrary-ID + auto-detection UX vLLM has and AIPerf lacks, on top
of a transport layer that already exceeds vLLM's. The genuinely new code is the
detection heuristic plus CLI/YAML plumbing that skips the catalog.

## Source anchors

- `rust/runtime/src/dataset/loader/public.rs` — HF fetch + composers; `HfAutoComposer`
  lands here.
- `rust/runtime/src/dataset/loader/mod.rs` — `DatasetSource`, `DatasetLoader`,
  `register_builtin_formats` (register `hf`), `probe`/`detect` (empty-probe constraint
  at :520).
- `rust/runtime/src/engine/execute/dataset_build.rs` — `build_public_dataset` lowering
  (accepts arbitrary source, no catalog).
- `rust/runtime/src/engine/dataset_input.rs` — `PublicDatasetSpec` /
  `PublicDatasetSourceSpec` wire shape (`deny_unknown_fields`).
- `rust/cli/src/load.rs` — `Inputs`, the dataset-kind selection chain (:931),
  `parse_dataset_filters` (:1681).
- `rust/cli/src/flags.rs` — new `--hf-*` flags.
- `rust/cli/src/yaml.rs` — `DatasetSection`, public-form parse + `ensure!` relaxation
  (:1266–1302).
- `rust/cli/resources/public_datasets.json` + `rust/cli/src/model/public_catalog.rs` —
  the catalog the passthrough intentionally bypasses (unchanged).
