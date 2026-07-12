# AIPerf-Rust: Dataset / Segment-Store / Loader Seam

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design
**Companions:** `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md`,
`2026-07-10-shared-rust-architecture-northstar.md`,
`2026-07-09-graph-ir-rust-port-design.md`,
`docs/reference/graph-segment-unified-store.md` (in the aiperf-graph-ir tree)

---

## 0. The one idea

The graph-IR **segment store** and the legacy **multi-modal Conversation/Turn mmap
cache** are the same problem solved twice. Unify them: one **content-addressed,
in-memory segment/blob store** with dense integer handles, and make
`Conversation`/`Turn` carry **handles, not bytes**. Then:

- The whole `memory_map_*` layer (~600 LOC), the backing-store/client-store split,
  the `DatasetClientMetadata` tagged union, the ZMQ conversation-fetch REQ/REP, the
  1 Hz rebroadcast, the gc-eviction dance, and `copy_with_stripped_media()` all
  **collapse to `Arc<SegmentStore>` + `Arc<[Conversation]>`**. (All confirmed
  accidental — they exist only to move a dataset across process boundaries.)
- The heavy/light split (`Turn` bytes vs media-free `TurnMetadata`) becomes
  **automatic**: metadata already carries no bytes; now the "heavy" `Conversation`
  also carries no bytes — only handles — so the bytes live exactly once, in the
  store.
- **Dedup + KV-cache prefix-reuse reasoning fall out for free** because media,
  text, and raw-payload all share one content-addressed id space (the property the
  graph segment store already exploits for >1M-rps synthetic runs).

This is redo-cleaner, not port-exact. The graph-IR store already proves the seam;
we promote it from a graph-only optimization to the universal dataset substrate.

---

## 1. What survives, what dies

**Essential domain logic — port (keep in spirit):**
- Conversation / Turn / Metadata model, incl. the media-free `*Metadata` projection
  (`dataset_models.py:184-373`) — the timing layer wants shape, dispatch wants bytes.
- The loader format zoo (synthetic, single/multi-turn jsonl, random_pool,
  mooncake/bailian/burst_gpt trace, dag_jsonl, raw_payload, sharegpt, exgentic, HF
  public, accuracy) — each is real parsing of a real format.
- Composition: turn finalization, ISL/OSL sequence-distribution sampling, context
  injection, model selection, max_tokens (`composer/base.py:28-208`).
- Sampling strategies (random/sequential/shuffle, deterministic seed) — already
  operate on `list[str]` ids, already process-agnostic (`dataset_samplers.py:32-87`).
- Tokenization (tokenize-at-load, so segment ids can be token-keyed).
- The graph segment store's **prefix-dependent content addressing** (parent id
  folded into the hash) and **splice materialization** (static segment + dynamic
  predecessor-reply interleave).

**Accidental complexity — delete (multiprocess-only):**
- `memory_map_{store,client,models,utils}.py`, `ConversationOffset`/
  `MemoryMapDatasetIndex`, per-fetch JSON decode, executor-thread read wrapping.
- `dataset_backing_store` + `dataset_client_store` plugin categories (one impl
  each) → one store.
- `DatasetClientMetadata` tagged union + its Pydantic core-schema shim
  (`dataset_models.py:69-134`).
- `CONVERSATION_REQUEST` / `CONVERSATION_TURN_REQUEST` ZMQ handlers
  (`dataset_manager.py:436-476`) and the worker's fetch fallback
  (`worker.py:1226-1238`) → a direct `dataset.get(id)`.
- `DatasetManager` evicting its own in-memory dataset then re-reading it through
  mmap (`dataset_manager.py:176-190`) → no-op.
- zstd `compress_only` + per-pod HTTP download → only re-appears behind the
  multi-node trait boundary (YAGNI now; see northstar).
- 1 Hz `DatasetConfiguredNotification` rebroadcast (ZMQ late-joiner workaround).

---

## 2. The segment/blob store seam (Rust)

Take tree-1's `aiperf-graph` seam (`crates/aiperf-graph/src/segment.rs`,
`materialize.rs`) as the base, add the production interned-handle form from the
Python unified store (`graph_segment_unified_store.py`), and generalize it to carry
media blobs too.

```rust
/// Opaque, deterministic, prefix-dependent content-address handle.
/// Dense u32 index into the store's interned arena (the >1M-rps form:
/// materialization is index -> Bytes slice, never a re-serialize/re-parse).
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Handle(u32);

pub enum Payload {
    /// Tokenized text/message segment. Id keyed on tokens (blake3 over
    /// parent \0 role \0 token-le-bytes) to skip re-tokenization at scale.
    Message { role: Role, wire: Bytes },   // pre-serialized OpenAI message
    /// Verbatim raw wire (raw_payload / raw_messages). Id keyed on wire_json
    /// (key-order-sensitive) so byte-exact replay survives.
    Raw { wire: Bytes },
    /// Multimodal blob (image/audio/video base64 or bytes). Id keyed on content.
    Media { kind: MediaKind, bytes: Bytes },
}

pub trait SegmentStore {
    fn get(&self, h: Handle) -> &Payload;
    /// Assemble a wire body by concatenating raw slices — zero JSON parse.
    fn build_body(&self, handles: &[Handle], overrides: &Overrides) -> Bytes;
}

pub struct SegmentPool { /* arena: Vec<Payload>; ids: HashMap<SegId, Handle> */ }
impl SegmentPool {
    /// Write seam: intern once, dedup by content id, return dense handle.
    pub fn intern(&mut self, parent: Option<Handle>, p: Payload) -> Handle;
}
```

**Invariants (all three existing implementations preserve these — keep them):**
1. **Ids are opaque, deterministic, prefix-dependent content hashes.** Parent id
   is folded into the hash, so identical text under different prefixes gets distinct
   ids and shared prefixes dedup to one — the basis for both cross-instance dedup
   and downstream KV-cache prefix-reuse reasoning.
2. **Distinct domain tags per id kind** (`message` / `raw` / `media` /
   `text-only`) so a token-keyed id can never alias a wire-keyed id.
   (`segment_ir/pool.py:37,62,81` uses three separate hash domains — replicate.)
3. **Materialize = clone/concat, never re-serialize.** The fast path
   (`build_body`) concatenates pre-serialized `Bytes` slices with only the
   per-dispatch overrides (max_tokens/model/stream) spliced in.
4. **Handles are dense integers**, not 32-hex strings, for the production form —
   the string form (segment.rs:91) is fine for a first cut but the `u32` +
   arena is the zero-copy scale form (`graph_segment_unified_store.py:162,497`).

**Materialization with dynamic splices** (DAG / multi-turn): port
`PromptMaterializer` / `SegmentItemsMaterializer` (`materialize.rs:22-54`). A node's
assembly program interleaves `Item::Seg(handle)` (static, prefix-cached) with
`Item::Splice(channel_ref)` (a predecessor node's captured reply, read live from
the run's channel state). This is what lets multi-turn history and DAG fork-context
reach the wire without re-materializing the static prefix each turn.

---

## 3. Conversation / Turn as handle-carriers

```rust
pub struct Turn {
    pub role: Role,
    pub model: Option<ModelId>,
    pub max_tokens: Option<u32>,
    pub timestamp: Option<i64>,
    pub delay: Option<i64>,
    // handles into the SegmentStore — NO bytes here
    pub messages: SmallVec<[Handle; 1]>,   // text/message segments
    pub media:    SmallVec<[Handle; 0]>,   // image/audio/video blobs
    pub raw:      Option<Handle>,          // verbatim raw_payload / raw_messages
    pub tools:    Option<Handle>,          // raw_tools (walks history)
    pub extra_body: Option<Bytes>,         // small; inline is fine
    // DAG authoring
    pub prerequisites: SmallVec<[NodeId; 0]>,
    pub branch_ids: SmallVec<[BranchId; 0]>,
}

pub struct Conversation {
    pub session_id: SessionId,
    pub turns: Vec<Turn>,
    pub system: Option<Handle>,
    pub user_context: Option<Handle>,
    pub accuracy: Option<AccuracyGroundTruth>,   // ground_truth + task
    pub dag: Option<DagMeta>,                     // agent_depth/branches/parent
}
```

`copy_with_stripped_media()` **disappears**: `Turn` never holds bytes, so the
control-plane view is just `Turn` without resolving handles, and the media-free
`TurnMetadata` projection is a trivial borrow. `Conversation::metadata()` stays as
the light projection the sampler/timing layer consumes.

---

## 4. Cleaner dataset-loader architecture (five stages, one direction)

Replace the composer→loader→backing-store→client-store→sampler tangle (whose
shape is dictated by process boundaries) with a linear pipeline:

```
 LOAD ──▶ COMPOSE ──▶ STORE ──▶ SAMPLE ──▶ MATERIALIZE
(format)  (finalize)  (intern)  (order)    (dispatch)
```

1. **Load** — `trait DatasetLoader { async fn load(&self, cfg) -> Vec<RawRow>; fn can_load(&self, src) -> bool; }`.
   One impl per format (the zoo). `can_load` auto-detection stays as a registry
   probe. Pure parse/fetch; no store coupling. (Essential — keep all of them.)
2. **Compose** — `trait Composer { fn compose(rows, cfg, tok, &mut SegmentPool) -> Vec<Conversation>; }`.
   Owns synthesis (prompt/image/audio/video generators), ISL/OSL
   sequence-distribution sampling, context injection, model selection,
   max_tokens finalization, tokenization. **Interns every text/media/raw blob into
   the pool as it builds each `Turn`**, so composition and content-addressing
   happen in one pass.
3. **Store** — the `SegmentPool` + `Dataset { conversations: Arc<[Conversation]>,
   index: HashMap<SessionId, usize>, segments: Arc<SegmentStore>, meta:
   DatasetMetadata }`. `Arc`-shared to every worker task. Replaces mmap +
   backing/client store entirely. Insertion order preserved by the `Vec` +
   `index` (the protocol guarantee `protocols.py:116-123` demanded).
4. **Sample** — `trait Sampler { fn next(&mut self) -> SessionId; }` over
   `ConversationMetadata` (media-free), deterministic seed. Unchanged in spirit.
5. **Materialize** — at dispatch, `segments.build_body(turn.handles(), overrides)`
   → `Bytes`, or the endpoint's `format_payload` when not a verbatim/raw turn.
   Splice-resolves DAG predecessor replies from run channel state.

**Accuracy note (ties to the port ledger §accuracy):** the accuracy loader stamps
`ground_truth`/`task` onto `Conversation` — keep that, but carry a **real
correlation id** from conversation → grading result, replacing the fragile
`session_num % len(tasks)` positional mapping. This is the same id that lets the
proposed `AccuracyAccumulator` associate a response with its ground truth typed,
not positionally.

---

## 5. Why this is the right seam

- **One store, one id space.** Text, media, and raw payloads are all
  content-addressed handles; dedup, prefix-reuse, and zero-parse body build are
  properties of the substrate, not per-feature code.
- **The heavy/light split is structural, not a `copy_with_stripped_media` band-aid.**
- **Multi-worker sharing is `Arc`, not a serialize→disk→mmap→deserialize round
  trip.** The single biggest wart (DatasetManager evicting RAM it already holds,
  only to re-read it through mmap) becomes a non-event.
- **The graph and non-graph paths converge.** Today the segment store is graph-IR
  only and the general path re-implements media handling on `Turn`. One substrate
  serves both — and the DAG splice materializer is just the multi-turn case of the
  same `PromptMaterializer`.
- **Multi-node stays behind a trait.** If distribution is ever needed, the
  interned-handle + `Bytes`-slice store is exactly what serializes efficiently
  (the Python unified store already spills to `content.blob` + `content.idx` span
  tables); but that lives behind the store trait, added only if needed (YAGNI).

---

## 6. Open decisions (flag before building)

1. **Handle width / arena eviction.** `u32` handles cap at 4B unique segments —
   fine. Do we ever evict (streaming datasets larger than RAM), or is
   whole-dataset-in-RAM the contract? Current Python already assumes the latter
   (it holds the full dict). Recommend: whole-dataset-in-RAM; revisit only for
   trace files that exceed memory.
2. **Text interning granularity.** Intern whole messages, or sub-message spans
   (LCP-trie lowering, per `2026-06-28-weka-segment-trie-ir-design.md`)? Whole
   message is simpler and enough for dedup; the trie buys finer KV-prefix modeling.
   Recommend: whole-message v1, trie behind the same `intern` seam later.
3. **When to tokenize.** Tokenize-at-compose (so ids are token-keyed and OSL/ISL
   are known) vs lazy. Recommend: at-compose, matching current behavior.
4. **`build_body` override splicing.** Overrides (max_tokens/model/stream) must
   splice into a pre-serialized body without a full parse — port the Python
   `build_request_body_handles(overrides_inner)` approach
   (`graph_segment_unified_store.py:497`).

## Addendum — 2026-07-11

The `Turn` / `Conversation` sketch intentionally shows the storage seam, but it omits
several Python dataset fields that affect dispatch, metrics, or context
reconstruction and therefore must be accounted for before implementation. The Rust
loader/model design must carry or deliberately lower: raw request payload/message
forms (`raw_payload`, `raw_messages`), raw tool definitions and tool-walk metadata,
per-turn extra headers/body/request parameters, audio duration, context-mode fields,
and endpoint/model overrides that influence wire formatting.

For graph/agentic inputs, keep the DAG metadata needed by scheduling and reporting:
branch ids, root/parent conversation ids, agent depth, fork/branch projections such as
`has_forks`, and any fields used to rebuild predecessor context. These can still be
stored as content-addressed handles where appropriate, but dropping them from the
schema would silently break dispatch parity, ASR/accuracy metrics, or replay/context
reconstruction.

## Addendum — 2026-07-11: implemented end to end

This addendum is authoritative for implementation status. The design is realized as
the `aiperf-dataset` crate and is shared by the native CLI and `aiperf-graph`; there is
no mmap/backing-store/client-store or graph-private segment-store fallback.

The five stages in §4 are concrete:

- **Load:** object-safe `DatasetLoader` implementations are paired with `Composer`s in
  `LoaderRegistry`, including `synthetic`, `synthetic_rankings`, `single_turn`,
  `multi_turn`, `random_pool`, `mooncake_trace`, `bailian_trace`, `burst_gpt`,
  `sagemaker_data_capture`, `dag_jsonl`, `raw_payload`, `inputs_json`, `sharegpt`,
  `exgentic`, `exgentic_v2`, `hf_asr`, `hf_instruction_response`, `hf_conversation`,
  `mt_bench`, `mmvu`, `spec_bench`, `speed_bench`, and `accuracy`. Local JSON/JSONL,
  CSV, Parquet, generic URL, Hugging Face Dataset Viewer, and immutable-revision Hub
  artifacts are covered. Remote fetch/cache is the injectable `DatasetFetcher` over
  the Clock-injected native transport. Decoded HF image/audio/video values are
  normalized at compose time rather than dropped.
- **Compose/store:** `Composer`, `TextTokenizer`, `MediaResolver`, `PromptGenerator`,
  `SyntheticMediaGenerator`, and `ModelSelector` are injectable traits. Composition
  interns tokenized message/text, byte-exact raw JSON, and media into a `SegmentPool`;
  `Dataset` freezes `Arc<[Conversation]>`, an insertion-order lookup index, and one
  `Arc<dyn SegmentStore>`. `Conversation`/`Turn` retain only dense `Handle`s for large
  or wire-sensitive values, including messages, tools, extra body/headers/query
  parameters, raw payloads, and media. Accuracy uses a real `CorrelationId`; DAG
  branches, prerequisites, root/parent lineage, depth, and `has_forks` projections are
  validated before freeze.
- **Sample:** `Sampler` plus the injectable `SamplerFactory`/`SamplerRegistry` provide
  deterministic random, sequential, and shuffle strategies. Native online dataset
  sources now honor each loader's recorded strategy instead of hard-coding sequential
  order.
- **Materialize:** `RequestMaterializer` and `EndpointResolver` reconstruct every
  context mode, retain live assistant replies, walk backward to the latest tool set,
  resolve endpoint/model/stream/header/query overrides, and propagate per-turn audio
  duration and accuracy identity. Raw bodies are byte-identical without overrides;
  explicit overrides append only a serialized object tail. Structured requests use the
  endpoint trait and derive accounting metadata from the final body so transport mode,
  wire fields, and reported model/OSL cannot drift.
- **Graph:** `aiperf-graph::SegmentItemsMaterializer` accepts the same
  `Arc<dyn SegmentStore>` and interleaves static handles with retained dynamic reply
  wires. The previous graph-local store and unused JSON `segment_pool` sentinel are
  deleted.

The four open decisions in §6 are resolved:

1. Handles are dense `u32`; the contract is whole-dataset-in-memory with no eviction.
2. V1 interns whole messages/text items. Alternate granularity remains implementable
   behind `SegmentStore`/the write-side interner without changing conversations.
3. Tokenization happens during composition through local tiktoken or Hugging Face
   tokenizer implementations; token IDs are authoritative for message/text identity.
4. Raw-object overrides use closing-brace tail splicing and never parse or rewrite the
   authored object. Message assembly clones/concatenates pre-serialized `Bytes` slices;
   dynamic graph replies retain their encoded wire representation.

Proof is executable and self-contained: `cargo test -p aiperf-dataset --all-targets`
passes 78 tests; `cargo clippy -p aiperf-dataset --all-targets -- -D warnings` is clean;
the graph suite passes 22 unit tests plus both real h1/h2c transport integrations; and
`cargo test -p aiperf --all-targets` passes the native runtime/CLI suite. In particular,
`crates/aiperf/tests/dataset_online.rs` starts a real loopback HTTP server, runs the
compiled CLI over a two-turn native dataset, and proves that the live first reply is
present in the second request. The dataset suite also executes FFmpeg-backed audio/video
generation and ASR normalization, exact raw replay, all four context modes, loader
auto-detection, fixed/prefix hashing, sampler reproducibility, Hugging Face pagination
and revision pinning, and endpoint-specific request construction.

This addendum completes the dataset/segment seam itself. The in-process offline engine
sink and execution of authored DAG branch policy are owned by their separate companion
specifications; dataset storage, validation, sampling, and materialization already
remain clock/backend-neutral for those consumers.

## Addendum — 2026-07-11 (opaque external-evaluator association)

The accuracy ground-truth storage described in §4 and the earlier implementation
addendum is superseded by the canonical Python/Lighteval evaluator boundary.
Rust dataset metadata must not carry an expected answer or hidden test payload.

`Conversation::accuracy` now contains only `AccuracyAssociation {
correlation_id, task }`. The correlation ID is the evaluator's opaque problem ID
and is propagated through ordinary request materialization so completed text can
be returned without positional matching. Prompt/messages and generation controls
remain normal segment-backed turn data. Ground truth stays inside the evaluator
worker and is never part of the Rust evaluator protocol or dataset metadata.

## Addendum — 2026-07-12 (token-native turn representation)

The no-mmap decision remains authoritative. The native equivalent of the
prototype's mmap-serialized token list is a `Payload::TokenIds` value in the
single content-addressed segment arena, referenced by `Turn::raw_token_ids`.
Token IDs are validated as a non-empty `u32` sequence during loading/composition;
`Dataset::validate_for_endpoint` then enforces any open endpoint descriptor whose
`requires_raw_token_ids` capability is true before scheduling starts.

`single_turn` accepts the canonical `raw_token_ids` field, the `token_ids` alias,
and the PR-1113-compatible `extra.token_ids` form; token-native rows validate
their nested sampling limit and normalize it as the effective output cap.
`raw_payload`/`inputs_json`
extract a top-level token array once; for a token-native endpoint they retain the
typed model/generation/extra fields, reject malformed canonical model, stream,
sampling, or output-limit values, and discard the original payload bytes.
Ordinary endpoints retain byte-exact raw replay unchanged. Synthetic composition
calls `PromptGenerator::generate_token_ids`, never decodes temporary text, and
replaces EOS with `(eos + 1) % vocab_size`, porting
`src/aiperf/dataset/generator/prompt.py:152-200` and
`src/aiperf/dataset/composer/synthetic.py:143-164` from
`ajc/in-engine-transport`.

Online materialization resolves the handle only when the prepared endpoint builds
its JSON body. The simulator-aware materializer emits no body for raw-token turns;
the feature-gated Dynamo adapter resolves the same handle directly. This extends,
rather than forks, the existing trace-hash native-materialization path.
