<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Baseten columnar-load performance

## Status

Approved design for the native Rust port of origin/main commit
`1e32a51318af3474dc1672d225fd495471ab45df`.

## Problem and measured evidence

The native Baseten loader currently uses Parquet's row API to decode every
column, converts every full row to `serde_json::Value`, collects the complete
file, reparses each value into `BasetenRow`, and then constructs the
loader/composer intermediate value twice. This makes unused wide columns part
of both decode cost and peak memory.

The existing release `dataset_load_bench` adapter, run with
`AIPERF_DATASET_BASETEN_SESSION_COLUMN=poor_man_session_id`, measured the real native
load -> compose path on a deterministic 100,000-row Baseten fixture. The file
is 177 MiB on disk and 190.8 MB as a decoded Arrow table; its replay columns are
accompanied by three unused 512-byte strings and other unused outcome fields.
Three warm-cache baseline runs reported:

| Run | Timed load -> compose | Peak RSS |
| ---: | ---: | ---: |
| 1 | 3.249 s | 562,772 KiB |
| 2 | 3.310 s | 564,064 KiB |
| 3 | 3.266 s | 562,604 KiB |

The row, conversation, turn, and input-token results were identical in all
runs: 100,000 rows, 50,000 conversations, 100,000 turns, and 6,400,000 input
tokens. This establishes an applicable native performance gap rather than a
source-only resemblance.

## Goals

1. Decode local Baseten Parquet and Arrow IPC files in bounded record batches.
2. Project only columns needed by the selected replay mode.
3. Select and sample sessions from metadata columns before allocating prompts.
4. Retain all existing native replay timing, filtering, KV-hint, grouping, and
   deterministic sampling behavior except the upstream-authored session-column
   policy described below.
5. Materially reduce measured load time and peak RSS on the pinned wide fixture.

## Non-goals

- Streaming the final dataset through execution. The shared `DatasetLoader`
  contract returns `Vec<RawRow>` and the resident `Dataset` is intentionally
  materialized for sampling and repeated phases.
- Porting arbitrary recorded outcome fields. That is tracker #40
  (`215be05b6a`) and must not be mixed into this performance change.
- Changing remote URL or Hugging Face acquisition; those sources continue
  through the shared public loader because their local artifact boundary is
  owned there.
- Adding a new columnar dependency. The existing `arrow` and `parquet`
  dependencies provide the readers; Arrow IPC is enabled on the existing
  optional Arrow dependency under the existing `parquet` feature.
- Generating or caching Python memory-map files. Native execution has its own
  resident-dataset and segment-store lifecycle.

## Source and format contract

Local `.parquet`, `.arrow`, and `.ipc` files are accepted when their schema
contains `timestamp_start_unix_ms`, `prompt`, `input_tokens`, and
`output_tokens`. Detection reads only file metadata.

The preferred session column is read from
`AIPERF_DATASET_BASETEN_SESSION_COLUMN`, defaulting to
`provided_session_id`. Accepted values are exactly `provided_session_id` and
`poor_man_session_id`. When the preferred column is absent but the other is
present, the loader falls back to the available column. When neither is
present, every retained row becomes a generated single-turn session. An
invalid environment value is a validation error during load; format detection
remains side-effect free.

This intentionally replaces native's repeated-group scoring with upstream's
schema/config policy. It avoids a full two-session-column scoring pass and
makes loader grouping agree with the merged Python resolver/count path.

## Columnar scan design

A private source object opens the file once and retains the schema. Each scan
uses a cloned descriptor and yields record batches of at most 128 rows, matching
the upstream batch bound. Parquet uses
`ParquetRecordBatchReaderBuilder` plus `ProjectionMask`; Arrow IPC uses
`arrow::ipc::reader::FileReader` with projected column indices.

The metadata scan projects the timestamp and selected session column. It
computes the minimum timestamp, per-session first timestamp, and null-session
row count. When no session sampling is requested, only the minimum is needed.
When sampling is active, sessions are ordered by `(first_timestamp,
session_id)` and sampled with the existing derived Python-compatible RNG. At
least one non-null session is retained when any exist, and null-session rows
are sampled by stable source ordinal.

The trace scan projects:

- the four required columns;
- the selected session column, when present;
- `total_hashes` and `block_size` unless `omit_kv_hints` is true;
- `duration_e2e_ms` only for closed-loop replay.

Rows rejected by session sampling are skipped before prompt/string allocation.
Each retained row is decoded directly into `BasetenRow`, normalized, filtered,
and converted to exactly one `Value`; that same value is serialized for
`RawRow::wire`. `RawRow::group_key` carries the selected session id so the
composer does not rescan or infer session policy from filtered rows.

## Error handling and resource bounds

File open, metadata, batch, projection, type, nullability, and scalar range
failures become `DatasetError::Validation` values containing the source path,
column, and stable row ordinal. Production code does not panic or silently
coerce malformed required fields.

Record-batch decode memory is bounded by 128 projected rows. Total memory is
still O(retained rows + selected session ids), as required by the public loader
and resident dataset contracts, but it is not O(all source columns) and no
second whole-file generic JSON vector exists.

## Upstream test mapping

| Upstream behavior | Native evidence |
| --- | --- |
| Arrow `.arrow` / `.ipc` parity with Parquet | Rust loader test builds both formats and compares composed conversations/turns |
| Arrow/Parquet schema detection | Rust `can_load` tests for all suffixes and missing required columns |
| Bounded projected batches | Rust integration fixture includes large unused columns; a scan observer test pins projected names and maximum batch size |
| Configured session column with fallback | Rust grouping tests cover preferred, fallback, and absent columns |
| Sampling metadata uses bounded scans and grouping reuses its key | Rust deterministic whole-session sampling test asserts no session shredding |
| Malformed required field after the initial rows still errors | Rust batch fixture places a null/negative required value after row 10 |
| Null hashes normalize to empty | Existing/new Rust row decode test preserves this behavior |
| Baseten skips redundant `inputs.json` | Existing native fixed-schedule gate in `engine/execute/compose_sidecars.rs` suppresses up-front inputs when a phase filters the dataset; no duplicate Baseten-specific gate is added |
| Python cache/plugin/docs changes | Exact two-parent merge `da917561fb` retains those upstream files; there is no Rust memory-map cache equivalent |

The upstream commit contains unit tests but no integration or end-to-end test.
Native additionally exercises the public `dataset_load_bench` adapter so the
release loader -> composer -> tokenizer boundary is measured.

## Acceptance criteria

1. Focused Baseten unit tests pass with `parquet` enabled and demonstrate a
   red-green cycle for Arrow detection/load and projection/sampling behavior.
2. The dataset-load integration tests cover the public Baseten adapter for both
   Parquet and Arrow IPC.
3. On the pinned 100,000-row fixture and identical release binary invocation,
   median timed load -> compose and median peak RSS are both lower than the
   three-run baseline; the exact counts remain unchanged.
4. `cargo fmt --check`, focused runtime tests, runtime Clippy with the relevant
   features, and a release benchmark build pass using `sccache` and the
   dedicated `/mnt/4tb` target.
5. The final Graham review reports no unresolved Critical or Important finding.
