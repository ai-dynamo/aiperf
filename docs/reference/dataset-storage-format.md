---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Dataset Storage Format
---

# Dataset Storage Format

AIPerf serializes every loaded dataset to a memory-mapped file before the benchmark starts. The storage format dictates what the worker reads at dispatch time and whether the wire payload needs to ride the records-pipeline ZMQ hop inline.

Two formats:

- **`PAYLOAD_BYTES`** — the wire payload for each turn is pre-encoded once at config time and written to mmap as raw bytes. Workers read the bytes and ship them verbatim to the inference server. The records pipeline reads the same bytes via its own mmap client; `MetricInputs.payload_bytes` is `None` on the wire — nothing payload-shaped crosses the records-pipeline ZMQ hop. This is the fast path.

- **`CONVERSATION`** — the full `Conversation` (turns, system message, user-context message, branches, metadata) is serialized as JSON in mmap. Workers deserialize the conversation per session and call `endpoint.format_payload(...)` at dispatch time to build the wire payload. The worker then puts the resulting bytes on `MetricInputs.payload_bytes` and the records pipeline reads them inline from the wire (msgpack `bin` span). This is the slow path — every request pays a per-dispatch encode.

Format is selected **per dataset, not per conversation**: every conversation in a single benchmark run uses the same format. The selector is `DatasetManager._select_mmap_format(conversations)` and it picks `PAYLOAD_BYTES` only when every turn in every conversation has its `raw_payload` field populated.

## How `raw_payload` gets populated

Either the loader writes it directly, or the composer pre-formats at config time:

- **Loader-direct**:
  - `raw_payload` loader: each JSONL line becomes `Turn(raw_payload=...)`.
  - `inputs_json` loader: each `payloads[]` entry becomes `Turn(raw_payload=...)`.
  - `mooncake_trace` in `payload` mode: `_build_turn` returns `Turn(raw_payload=trace.payload, ...)`.
- **Composer pre-formatting** (`BaseDatasetComposer._preformat_payloads`): for any conversation where the wire payload is fully determined at config time, the composer walks every turn and stamps `turn.raw_payload = endpoint.format_payload(request_info)` at load time. This is how `mooncake_trace` in `messages` mode (loader sets `raw_messages`), synthetic chat / image / rankings, public HF datasets, `random_pool`, and `single_turn` end up with `raw_payload` on every turn — even though their loaders never touched the field.

After either path, `_select_mmap_format` picks `PAYLOAD_BYTES` and the runtime treats the dataset like an explicit replay dataset.

## Which dataset shapes pick which format

| Dataset shape | Format | Why |
|---|---|---|
| `raw_payload` (single-file or directory mode) | `PAYLOAD_BYTES` | Loader stamps `raw_payload` on every turn; multi-turn pre-canned via `MESSAGE_ARRAY_WITH_RESPONSES` context mode |
| `inputs_json` | `PAYLOAD_BYTES` | Loader stamps `raw_payload` on every turn (one per `payloads[]` entry) |
| `mooncake_trace` (`payload` or `messages` mode) | `PAYLOAD_BYTES` | Loader sets the turn body directly; `_infer_context_mode` auto-detects `MESSAGE_ARRAY_WITH_RESPONSES` for self-contained traces |
| `mooncake_trace` synthesized, single-turn | `PAYLOAD_BYTES` | Composer pre-formats |
| `mooncake_trace` synthesized, multi-turn | `CONVERSATION` | `context_mode` defaults to `None` (treated as `DELTAS_WITHOUT_RESPONSES`); multi-turn under that mode requires live response capture between turns |
| `synthetic` chat, single-turn (`--num-turns 1`, the default) | `PAYLOAD_BYTES` | Composer pre-formats |
| `synthetic` chat, multi-turn (`--num-turns N > 1`) | `CONVERSATION` | Same reason as multi-turn synthesized mooncake — default `DELTAS_WITHOUT_RESPONSES` requires live response capture |
| `synthetic_rankings` | `PAYLOAD_BYTES` | Single-turn by construction; composer pre-formats |
| `synthetic` image / `image_retrieval` endpoint | `PAYLOAD_BYTES` | Single-turn by construction (`ImageRetrievalEndpoint.format_payload` rejects multi-turn); composer pre-formats |
| `public` / HF datasets (e.g. ShareGPT) | `PAYLOAD_BYTES` | Loaders filter to single-turn entries; composer pre-formats |
| `random_pool` | `PAYLOAD_BYTES` | Single-turn by construction; composer pre-formats |
| `accuracy` benchmarks | `PAYLOAD_BYTES` | `DatasetManager._load_accuracy_dataset` calls `preformat_payloads(...)` on the loader output directly (accuracy bypasses the composer chain), so every turn gets `raw_payload` populated just like a composer-fed dataset. |
| `dag_jsonl` with FORK branches | `CONVERSATION` (refused for `PAYLOAD_BYTES`) | FORK children seed from parent's accumulated turn list; structured turn replay is required. `_select_mmap_format` raises `ValueError` if any conversation has a FORK branch AND every turn has `raw_payload`; the composer's pre-format skips FORK conversations to avoid creating that combination. |
| `dag_jsonl` SPAWN-only (or any DAG child) | `CONVERSATION` | Composer's pre-format check rejects every non-root conversation (`is_root=False AND parent_conversation_id is not None`). Catches FORK children uniformly without needing per-parent branch-mode lookup. |
| Any dataset where the endpoint plugin can't format the turn (e.g. `RawEndpoint` over un-pre-encoded data) | `CONVERSATION` | Composer skips pre-format gracefully on `NotImplementedError` |
| Empty dataset (zero conversations or zero total turns) | `CONVERSATION` | `PAYLOAD_BYTES` would produce a zero-byte mmap file that the deserializer rejects with a misleading error |

## When you can't get `PAYLOAD_BYTES`

There are three structural reasons format selection falls back to `CONVERSATION`:

### 1. Live response capture between turns

When `context_mode` is `DELTAS_WITHOUT_RESPONSES` (the default for multi-turn synthetic and most HF datasets) and the conversation has more than one turn, the worker is supposed to capture each turn's live assistant response and fold it into the next turn's prompt. The wire payload for turn N depends on the runtime response from turn N−1, which is not knowable at config time.

If your benchmark needs multi-turn fidelity AND the `PAYLOAD_BYTES` fast path, switch the dataset's context mode to `MESSAGE_ARRAY_WITH_RESPONSES` or `DELTAS_WITH_RESPONSES` and include pre-canned assistant turns. See [Conversation Context Mode](conversation-context-mode.md) for the per-mode contract.

### 2. FORK branches in a DAG

FORK children inherit the parent's accumulated turn list at spawn time, which includes whatever live responses the parent captured before the fork point. Like #1, the child's payload depends on runtime state. `_select_mmap_format` refuses outright (raises `ValueError`) when raw-payload turns coexist with FORK branches; the composer's pre-format skips every DAG child (`is_root=False AND parent_conversation_id is not None`) for the same reason. The check is uniform on DAG children rather than per-branch-mode — simpler invariant, no cross-conversation lookup at composer time.

### 3. The endpoint plugin can't format the turn

The composer calls `endpoint.format_payload(request_info)` on each turn during pre-format. If the endpoint plugin raises `NotImplementedError` (the typical signal from `RawEndpoint` when no `raw_payload` is set), the composer logs an info message and falls back to `CONVERSATION` for the whole dataset. The benchmark still runs — worker dispatch handles formatting at runtime per turn — it just doesn't ride the fast path.

## What runs differently on each path

| Stage | `PAYLOAD_BYTES` | `CONVERSATION` |
|---|---|---|
| **Worker session type** | `RawPayloadSession` (conversation_id + num_turns + turn_index, no body) | `ContentSession` (full Conversation + turn_list + response storage + FORK seeding) |
| **Worker dispatch** | `dataset_client.get_payload_bytes(conv_id, turn_idx)` → wire | `endpoint.format_payload(request_info)` → wire |
| **`MetricInputs.payload_bytes` on the wire** | `None` (records-process resolves via its own mmap client) | inline (msgpack `bin` span, length-prefixed, no base64) |
| **Records-side parse** | `dataset_client.get_payload_bytes(conv_id, turn_idx)` → `orjson.loads(bytes)` once | `orjson.loads(mi.payload_bytes)` once |
| **`orjson.dumps` count per request** | 0 at runtime (bytes were produced once at load time by the backing store) | 1 (worker `format_payload` → dict → `orjson.dumps` → bytes; transport sends the bytes verbatim) |

## Backward compatibility

`MemoryMapFormat.CONVERSATION` is the conservative default. Any new loader or dataset type that doesn't explicitly populate `raw_payload` falls through to `CONVERSATION` automatically. Adding `PAYLOAD_BYTES` support to a new loader is opt-in: either populate `turn.raw_payload` from the source data, or let the composer pre-format pass do it. Either way the change is one-line and doesn't risk regressing existing datasets.

## Related

- [Conversation Context Mode](conversation-context-mode.md) — when a dataset is multi-turn-eligible for `PAYLOAD_BYTES` vs. requires `CONVERSATION` for live response capture.
- [Architecture](../architecture.md) — the broader record-pipeline data flow that consumes both formats uniformly downstream.
