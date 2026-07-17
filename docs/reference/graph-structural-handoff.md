# Graph Structural Sidecar Handoff (Phase 2)

This document describes the `graph_meta.msgpack` sidecar — the mandatory
artifact that hands a pre-built structural `ParsedGraph` from the DatasetManager
build plane to the TimingManager schedule plane, so the schedule plane never
re-parses the graph workload file (weka, dynamo, native, or dag_jsonl). The
DatasetManager writes it on every graph build route and advertises its exact
path on the graph-typed `DatasetConfiguredNotification`; the TimingManager
ingests it from that broadcast path only.

## Two-plane architecture

A graph IR workload is processed by two independent services in separate
processes:

| Plane | Service | Responsibility |
|-------|---------|----------------|
| Build | `DatasetManager` (build owned by `GraphStoreBuilder`, `src/aiperf/dataset/graph/store_build.py`) | Build the unified segment store via one of two drains — weka streams worker-parsed trace payloads and merges their per-trace structural graphs; dynamo/native/dag_jsonl parse ONCE in-process and drain that whole-graph parse through the interned builder — then write the mandatory sidecar (from the merged structural graphs on the weka route; DIRECTLY from the stripped parse, in parse order, on the non-weka route). The DatasetManager broadcasts the resulting store and sidecar paths on the graph-typed `DatasetConfiguredNotification` |
| Schedule | `TimingManager` | Ingest the sidecar from the broadcast path (hard fail if the broadcast is not graph-typed or the sidecar is unloadable), hand the `ParsedGraph` to `PhaseOrchestrator` |

The sidecar bridges these planes: the build plane writes it once and advertises
its location on the broadcast, the schedule plane reads it later from that exact
path — never a second parse of the workload.

```mermaid
sequenceDiagram
    participant DM as DatasetManager
    participant B as GraphStoreBuilder
    participant FS as Filesystem
    participant TM as TimingManager

    DM->>DM: validate_graph_endpoint_type(run)
    DM->>B: GraphStoreBuilder(run).build(graph_path)
    alt weka_trace (worker-pool payload stream)
        B->>B: stream worker-parsed payloads (pool)
        B->>FS: drain payloads into unified segment store
        B->>B: _merge_structural_graphs(structural_sink) → merged structural ParsedGraph
    else dynamo / native / dag_jsonl (in-process interned drain)
        B->>B: parse_graph_workload(run, path) → whole-graph ParsedGraph
        B->>FS: build_unified_trie_store_interned(parsed) → unified segment store
        B->>B: strip_replay_text(parsed) → structural ParsedGraph (parse order)
    end
    B->>B: catalogs_match(structural, store_catalog)?
    alt catalogs match
        B->>FS: write_graph_meta_sidecar() → graph_meta.msgpack
        B-->>DM: GraphStoreBuildResult(facet, sidecar_path, base_path)
        DM->>TM: DatasetConfiguredNotification(GraphSegmentClientMetadata: store + sidecar path)
    else catalogs diverge / empty or unmergeable stream
        B->>B: raise DatasetError (run fails; the sidecar is mandatory)
    end

    TM->>FS: read sidecar_path from GraphSegmentClientMetadata
    alt graph-typed, present, decodable, index cross-check passes
        FS-->>TM: graph_meta.msgpack bytes
        TM->>TM: decode_graph_meta_sidecar() → structural ParsedGraph
        TM->>TM: use sidecar graph (never re-parses)
    else not graph-typed / missing / undecodable / index-divergent
        TM->>TM: raise InvalidStateError (hard configure-time failure)
    end
```

## The `graph_meta.msgpack` sidecar

**File location:** `<MMAP_BASE_PATH>/aiperf_graph_meta_<benchmark_id>/graph_meta.msgpack`

resolved by `sidecar_path_for` (the unified segment store lives in its own
`aiperf_graph_segments_<benchmark_id>/` directory).

**Producer:** `GraphStoreBuilder._write_graph_sidecar`
(`src/aiperf/dataset/graph/store_build.py`), called on both build routes. The weka payload-stream drain writes it from the merged structural
graphs (see below). Every non-weka format (dynamo, native, dag_jsonl —
slot-free or slot-carrying alike) parses once in-process and takes the interned
drain, which writes the sidecar DIRECTLY from that whole-graph parse (its traces
stay in parse order). Slot-carrying graphs (live-reply dag_jsonl lineage, or
trie assembly items/capture from the native `@channel` lowering) ride this
non-weka route naturally — the interned drain is the only one that persists slot
envelopes, and weka corpora never carry slots. Either way the write is
mandatory: a catalog-divergent sidecar raises `DatasetError` (an unwritable
path fails the build with the underlying I/O error).

**Consumer:** `TimingManager._load_graph_sidecar` (ingests from the
`GraphSegmentClientMetadata` on the graph-typed `DatasetConfiguredNotification`).

**Content-free:** The sidecar encodes a **structural** `ParsedGraph` produced by
`strip_replay_text`: graph topology, node ids/types, edges, `arrival_offset_us`,
catalog keys, and the native dispatch fields (model / max_tokens / raw_tools /
extra_headers / theoretical prefix-cache counts) are preserved, but per-trace content is stripped: every
trace's `replay_outputs` (per-node recorded output channel values,
`node_id -> {channel: value}`) is cleared to empty. For the trie IR (`segment_pool is not None`) the strip is deeper:
the `SegmentPool` is emptied (kept non-`None` so the loaded graph still takes the
trie ordinal scheme), and each `LlmNode`'s inline `prompt` and `metadata["trie"]`
contents (`prompt_segment_ids`, raw `hash_ids`) are cleared — only the `"trie"`
marker key is kept. The real content lives in the unified segment store; the
sidecar never duplicates it.

**Wire format** (`encode_graph_meta_sidecar` / `decode_graph_meta_sidecar` in
`codecs.py`):

```
[header, pg_bytes]
```

where `header` is
`{"kind": "parsed_graph", "schema_version": int, "source_fingerprint": dict}`
and `pg_bytes` is the canonical `encode_parsed_graph_msgpack` output (a nested
blob). The `kind` discriminator is **required**: `decode_graph_meta_sidecar`
rejects any frame whose `kind` is missing or not `"parsed_graph"` (raising
`ValueError`, which the TimingManager surfaces as a hard `InvalidStateError`),
which is why the schema version was bumped from 2 to 3 — a pre-v3 kind-less
sidecar decodes as invalid and fails the run until it is rebuilt.
`GRAPH_META_SCHEMA_VERSION` is now 4: the 3→4 bump marks the verbatim
raw-JSON `Segment.wire_json` variant, but unlike 2→3 it adds **no reader-side
gate** — the version is advisory provenance only. A pre-v4 blob (which
normalized every segment to `{"role", "content"}`, dropping key order and extra
keys) still decodes fine because `Segment.wire_json` defaults to `None` (a
normalized role/content segment); only the 2→3 `kind`-required transition
gates. The outer
`_SIDECAR_DECODER` is untyped so the frame is decoded
without knowing the inner type; `pg_bytes` is then decoded by the typed
`_PG_MSGPACK_DECODER`.

## Promote-time catalog cross-check

Before writing the sidecar, `GraphStoreBuilder` verifies the structural graph's
catalog matches the store's. On the non-weka in-process interned drain (dynamo,
native, dag_jsonl) the structural graph comes from `strip_replay_text` on the
real-content `ParsedGraph` (the same object that built the unified store):

```python
structural = strip_replay_text(parsed)
if not catalogs_match(structural, catalog):
    raise DatasetError("graph_meta sidecar catalog mismatch: ...")
write_graph_meta_sidecar(...)
```

On the weka payload-stream drain the same gate runs against the merged
structural graph — assembled from the per-trace content-free graphs emitted
alongside the store payloads by the weka pool workers (each worker calls
`iter_trace_segment_payloads` on its parsed item).

`catalogs_match` (`graph_meta_sidecar.py`) calls `build_catalog_context` on the
structural graph and compares its `.catalog` dict to the content-build catalog.
Because both come from the same parse(s), a divergence would indicate a bug in
the strip/merge rather than a separate-parse race — the check is a safety
net. A mismatch raises `DatasetError` and fails the run: the sidecar is
mandatory, and one describing a different topology than the stored envelopes
would misschedule the workers.

## Load-time index cross-check

When the TimingManager loads an existing sidecar, `_sidecar_passes_index_check`
performs a best-effort verification against the unified store's per-node manifest
index (the store the worker will actually read):

```python
sidecar_matches_index(graph, index_offsets)
```

`sidecar_matches_index` (`graph_meta_sidecar.py`) is a pure comparison: it checks
that every node ordinal in the sidecar's per-trace catalog is present in the
supplied store-index ordinal set. A missing ordinal means the sidecar's topology
diverged from the stored manifests, so it returns `False` and the TimingManager
raises `InvalidStateError`. The surrounding `TimingManager._sidecar_passes_index_check`
handles reachability: any I/O failure while opening the unified store — including
an absent store — is treated as "not reachable" and returns `True`, so the
sidecar is accepted rather than triggering a spurious hard failure.

## Cache travel: retired

The cross-run graph cache (and with it the sidecar's promote/install
travel) was removed: every run builds its stores fresh, and the sidecar is
written beside them in the run's own benchmark directory. There is no
persistent cache for the sidecar to travel through.

## Build drains: weka payload stream vs non-weka interned

`GraphStoreBuilder._build_graph_store_streaming`
(`src/aiperf/dataset/graph/store_build.py`) dispatches on the workload format
to ONE of two drains; both build the SAME unified store and write the mandatory
sidecar, but they source the structural graph differently.

**Weka — worker-pool payload stream (`_build_graph_store_streaming_trie`).** Weka
sources — local `.json` file, directory of `.json` files, or HF corpus id —
stream worker-parsed payloads one trace at a time, so the parent never holds a
whole-corpus real-content `ParsedGraph`. Each streamed trace's content-free
structural graph is collected into a `structural_sink` and merged once
(`_merge_structural_graphs`, which raises `DatasetError` on an empty or
unmergeable stream); the merged structural graph feeds both the sidecar and the
per-node prefix-cache map. `_write_graph_sidecar` then writes `graph_meta.msgpack`
after the catalog cross-check (`catalogs_match`; a mismatch raises `DatasetError`).

**Every other format — in-process interned drain (`_build_interned_unified_store`).**
dynamo, native, and dag_jsonl parse once in-process (whole-graph) and drain that
SAME parse through `build_unified_trie_store_interned`, with no payload round
trip (in-process there is no worker pool to fan out to). The sidecar is written
DIRECTLY from the stripped whole parse (`_write_graph_sidecar(parsed, ...)`), so
its traces are in PARSE order (the weka merge sorts by id), and the full parse
is the per-node prefix-cache source. This drain is also the only one that
persists dynamic-slot envelopes (live-reply dag_jsonl lineage, native trie
assembly items/capture), so slot-carrying graphs ride it with no separate
fallback; `graph_carries_assembly_slots` no longer routes the store build (it
survives as the schedule-plane t\*-gate predicate).

## Key symbols

| Symbol | Location |
|--------|----------|
| `strip_replay_text` | `src/aiperf/dataset/graph/graph_meta_sidecar.py` |
| `write_graph_meta_sidecar` | `src/aiperf/dataset/graph/graph_meta_sidecar.py` |
| `sidecar_path_for` | `src/aiperf/dataset/graph/graph_meta_sidecar.py` |
| `catalogs_match` | `src/aiperf/dataset/graph/graph_meta_sidecar.py` |
| `sidecar_matches_index` | `src/aiperf/dataset/graph/graph_meta_sidecar.py` |
| `encode_graph_meta_sidecar` / `decode_graph_meta_sidecar` | `src/aiperf/dataset/graph/codecs.py` |
| `GRAPH_META_SIDECAR_FILENAME` | `src/aiperf/dataset/graph/codecs.py` |
| `_load_graph_sidecar` | `src/aiperf/timing/manager.py` |
| `_sidecar_passes_index_check` | `src/aiperf/timing/manager.py` |
| `GraphSegmentClientMetadata` / `GraphDatasetMetadata` | `src/aiperf/common/models/dataset_models.py` |
| `DatasetManager._configure_graph_dataset` / `_configure_graph_workload` / `_build_graph_store` | `src/aiperf/dataset/dataset_manager.py` |
| `GraphStoreBuilder` / `GraphStoreBuildResult` | `src/aiperf/dataset/graph/store_build.py` |
| `GraphStoreBuilder._write_graph_sidecar` | `src/aiperf/dataset/graph/store_build.py` |
| `GraphStoreBuilder._merge_structural_graphs` | `src/aiperf/dataset/graph/store_build.py` |
| `iter_trace_segment_payloads` | `src/aiperf/dataset/graph/segment_ir/store_builder.py` |
