# AIPerf-Rust: Dataset / Segment-Store / Loader Seam

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — realized as the `aiperf_runtime::dataset` module (loader → compose → dense-handle store → sampler → materializer), shared by `aiperf --execute`, evaluator-authored static accuracy, Graph-IR, and the library-only offline adapter.
**Companions:** `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md`,
`2026-07-10-shared-rust-architecture-northstar.md`,
`2026-07-09-graph-ir-rust-port-design.md`,
`docs/reference/graph-segment-unified-store.md` (in the aiperf-graph-ir tree)

---

## 0. The one idea

The graph-IR **segment store** and the legacy **multi-modal Conversation/Turn mmap
cache** are the same problem solved twice. They are unified: one **content-addressed,
in-memory segment/blob store** with dense integer handles, and `Conversation`/`Turn`
carry **handles, not bytes**. As a result:

- The whole `memory_map_*` layer (~600 LOC), the backing-store/client-store split,
  the `DatasetClientMetadata` tagged union, the ZMQ conversation-fetch REQ/REP, the
  1 Hz rebroadcast, the gc-eviction dance, and `copy_with_stripped_media()` all
  **collapse to `Arc<dyn SegmentStore>` + `Arc<[Conversation]>`**. (All confirmed
  accidental — they existed only to move a dataset across process boundaries.)
- The heavy/light split (`Turn` bytes vs media-free `TurnMetadata`) is
  **automatic**: metadata already carries no bytes; now the "heavy" `Conversation`
  also carries no bytes — only handles — so the bytes live exactly once, in the
  store.
- **Dedup + KV-cache prefix-reuse reasoning fall out for free** because media,
  text, raw-payload, and raw token arrays all share one content-addressed id space
  (the property the graph segment store already exploits for >1M-rps synthetic runs).

This is redo-cleaner, not port-exact. The graph-IR store proved the seam; it is
promoted from a graph-only optimization to the universal dataset substrate, with no
mmap/backing-store/client-store or graph-private segment-store fallback remaining.

---

## 1. What survives, what dies

**Essential domain logic — kept (in spirit):**
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

**Accidental complexity — deleted (multiprocess-only):**
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

The seam takes the `aiperf_runtime::graph` module's base (`rust/runtime/src/graph/segment.rs`,
`materialize.rs`), adds the production interned-handle form from the Python unified
store (`graph_segment_unified_store.py`), and generalizes it to carry media blobs
and raw token arrays too.

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
    /// Validated raw token array. Id keyed on the u32 token sequence; the
    /// native equivalent of the prototype's mmap-serialized token list.
    TokenIds { tokens: Vec<u32> },
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

**Invariants (all implementations preserve these):**
1. **Ids are opaque, deterministic, prefix-dependent content hashes.** Parent id
   is folded into the hash, so identical text under different prefixes gets distinct
   ids and shared prefixes dedup to one — the basis for both cross-instance dedup
   and downstream KV-cache prefix-reuse reasoning.
2. **Distinct domain tags per id kind** (`message` / `raw` / `media` /
   `token-ids` / `text-only`) so a token-keyed id can never alias a wire-keyed id.
   (`segment_ir/pool.py:37,62,81` uses separate hash domains — replicated.)
3. **Materialize = clone/concat, never re-serialize.** The fast path
   (`build_body`) concatenates pre-serialized `Bytes` slices with only the
   per-dispatch overrides (max_tokens/model/stream) spliced in.
4. **Handles are dense integers** (`u32`), not 32-hex strings — the zero-copy scale
   form (`graph_segment_unified_store.py:162,497`).

**Materialization with dynamic splices** (DAG / multi-turn): `PromptMaterializer` /
`SegmentItemsMaterializer` (`materialize.rs:22-54`). A node's assembly program
interleaves `Item::Seg(handle)` (static, prefix-cached) with
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
    pub audio_duration: Option<f64>,
    // handles into the SegmentStore — NO bytes here
    pub messages: SmallVec<[Handle; 1]>,   // text/message segments
    pub media:    SmallVec<[Handle; 0]>,   // image/audio/video blobs
    pub raw:      Option<Handle>,          // verbatim raw_payload / raw_messages
    pub raw_token_ids: Option<Handle>,     // validated Payload::TokenIds array
    pub tools:    Option<Handle>,          // raw_tools (walks history)
    pub extra_body:    Option<Handle>,     // extra body / headers / query params
    // DAG authoring
    pub prerequisites: SmallVec<[NodeId; 0]>,
    pub branch_ids: SmallVec<[BranchId; 0]>,
}

pub struct Conversation {
    pub session_id: SessionId,
    pub turns: Vec<Turn>,
    pub system: Option<Handle>,
    pub user_context: Option<Handle>,
    pub accuracy: Option<AccuracyAssociation>,   // opaque correlation_id + task
    pub dag: Option<DagMeta>,                     // agent_depth/branches/parent
}
```

`copy_with_stripped_media()` **disappears**: `Turn` never holds bytes, so the
control-plane view is just `Turn` without resolving handles, and the media-free
`TurnMetadata` projection is a trivial borrow. `Conversation::metadata()` stays as
the light projection the sampler/timing layer consumes.

Beyond the storage seam, the model carries or deliberately lowers every Python
dataset field that affects dispatch, metrics, or context reconstruction: raw request
payload/message forms (`raw_payload`, `raw_messages`), raw tool definitions and
tool-walk metadata, per-turn extra headers/body/request parameters, audio duration,
context-mode fields, and endpoint/model overrides that influence wire formatting.
For graph/agentic inputs the DAG metadata needed by scheduling and reporting —
branch ids, root/parent conversation ids, agent depth, and fork/branch projections
such as `has_forks` — is retained (as content-addressed handles where appropriate)
so dispatch parity, ASR/accuracy metrics, and replay/context reconstruction do not
silently break.

**Accuracy association is opaque.** Rust dataset metadata carries no expected answer
or hidden-test payload. `Conversation::accuracy` holds only `AccuracyAssociation {
correlation_id, task }`, where `correlation_id` is the evaluator's opaque problem ID.
It is propagated through ordinary request materialization so completed text can be
returned to the evaluator without positional matching. Prompt/messages and generation
controls remain normal segment-backed turn data; ground truth stays inside the
canonical Python/Lighteval evaluator worker and is never part of the Rust evaluator
protocol or dataset metadata. This replaces the fragile `session_num % len(tasks)`
positional mapping and the earlier Rust-held `ground_truth`/`task` sketch.

---

## 4. The dataset-loader architecture (five stages, one direction)

The composer→loader→backing-store→client-store→sampler tangle (whose shape was
dictated by process boundaries) is replaced by a linear pipeline:

```
 LOAD ──▶ COMPOSE ──▶ STORE ──▶ SAMPLE ──▶ MATERIALIZE
(format)  (finalize)  (intern)  (order)    (dispatch)
```

1. **Load** — object-safe `DatasetLoader` implementations are paired with `Composer`s
   in `LoaderRegistry`, one per format: `synthetic`, `synthetic_rankings`,
   `single_turn`, `multi_turn`, `random_pool`, `mooncake_trace`, `bailian_trace`,
   `burst_gpt`, `sagemaker_data_capture`, `dag_jsonl`, `raw_payload`, `inputs_json`,
   `sharegpt`, `exgentic`, `exgentic_v2`, `hf_asr`, `hf_instruction_response`,
   `hf_conversation`, `mt_bench`, `mmvu`, `spec_bench`, `speed_bench`, and `accuracy`.
   Local JSON/JSONL, CSV, Parquet, generic URL, Hugging Face Dataset Viewer, and
   immutable-revision Hub artifacts are covered. Remote fetch/cache is the injectable
   `DatasetFetcher` over the Clock-injected native transport. `can_load`
   auto-detection stays as a registry probe. Decoded HF image/audio/video values are
   normalized at compose time rather than dropped.

   The `dag_jsonl` graph format bypasses the linear `Dataset`/`Conversation`/
   `DagMetadata` composition path: it enters the runner-owned graph-input resolver
   once, calls exactly one compiler, and never passes through a second registry.

2. **Compose** — `Composer`, `TextTokenizer`, `MediaResolver`, `PromptGenerator`,
   `SyntheticMediaGenerator`, and `ModelSelector` are injectable traits. Composition
   owns synthesis (prompt/image/audio/video generators), ISL/OSL
   sequence-distribution sampling, context injection, model selection, max_tokens
   finalization, and tokenization. It **interns tokenized message/text, byte-exact
   raw JSON, media, and validated raw token arrays into the `SegmentPool` as it
   builds each `Turn`**, so composition and content-addressing happen in one pass.
   Accuracy composition stamps a real `CorrelationId`; DAG branches, prerequisites,
   root/parent lineage, depth, and `has_forks` projections are validated before
   freeze.

3. **Store** — the `SegmentPool` + `Dataset { conversations: Arc<[Conversation]>,
   index: HashMap<SessionId, usize>, segments: Arc<dyn SegmentStore>, meta:
   DatasetMetadata }`. `Arc`-shared to every worker task; replaces mmap +
   backing/client store entirely. Insertion order is preserved by the `Vec` +
   `index` (the protocol guarantee `protocols.py:116-123` demanded).

4. **Sample** — `Sampler` plus the injectable `SamplerFactory`/`SamplerRegistry`
   provide deterministic random, sequential, and shuffle strategies over
   `ConversationMetadata` (media-free). Native online dataset sources honor each
   loader's recorded strategy instead of hard-coding sequential order.

5. **Materialize** — `RequestMaterializer` and `EndpointResolver` reconstruct every
   context mode, retain live assistant replies, walk backward to the latest tool set,
   resolve endpoint/model/stream/header/query overrides, and propagate per-turn audio
   duration and accuracy identity. Raw bodies are byte-identical without overrides;
   explicit overrides append only a serialized object tail (closing-brace tail
   splicing), never parsing or rewriting the authored object. Structured requests use
   the endpoint trait and derive accounting metadata from the final body so transport
   mode, wire fields, and reported model/OSL cannot drift. DAG predecessor replies are
   splice-resolved from run channel state.

---

## 4a. Token-native turn representation

Token-native inputs live in the same content-addressed arena rather than a separate
mmap-serialized token list. A `Payload::TokenIds` value is referenced by
`Turn::raw_token_ids`; token IDs are validated as a non-empty `u32` sequence during
loading/composition. `Dataset::validate_for_endpoint` enforces any open endpoint
descriptor whose `requires_raw_token_ids` capability is true before scheduling starts.

- `single_turn` accepts the canonical `raw_token_ids` field, the `token_ids` alias,
  and the PR-1113-compatible `extra.token_ids` form; token-native rows validate their
  nested sampling limit and normalize it as the effective output cap.
- `raw_payload`/`inputs_json` extract a top-level token array once; for a token-native
  endpoint they retain the typed model/generation/extra fields, reject malformed
  canonical model/stream/sampling/output-limit values, and discard the original
  payload bytes. Ordinary endpoints retain byte-exact raw replay unchanged.
- Synthetic composition calls `PromptGenerator::generate_token_ids`, never decodes
  temporary text, and replaces EOS with `(eos + 1) % vocab_size`, porting
  `src/aiperf/dataset/generator/prompt.py:152-200` and
  `src/aiperf/dataset/composer/synthetic.py:143-164`.

Online materialization resolves the handle only when the prepared endpoint builds its
JSON body. The simulator-aware materializer emits no body for raw-token turns; the
feature-gated Dynamo adapter resolves the same handle directly. This extends, rather
than forks, the existing trace-hash native-materialization path.

---

## 5. Why this is the right seam

- **One store, one id space.** Text, media, raw payloads, and raw token arrays are
  all content-addressed handles; dedup, prefix-reuse, and zero-parse body build are
  properties of the substrate, not per-feature code.
- **The heavy/light split is structural, not a `copy_with_stripped_media` band-aid.**
- **Multi-worker sharing is `Arc`, not a serialize→disk→mmap→deserialize round
  trip.** The single biggest wart (DatasetManager evicting RAM it already holds,
  only to re-read it through mmap) is a non-event.
- **The graph and non-graph paths converge.** `SegmentItemsMaterializer` accepts the
  same `Arc<dyn SegmentStore>` and interleaves static handles with retained dynamic
  reply wires; the previous graph-local store and unused JSON `segment_pool` sentinel
  are deleted. The DAG splice materializer is just the multi-turn case of the same
  `PromptMaterializer`.
- **Multi-node stays behind a trait.** If distribution is ever needed, the
  interned-handle + `Bytes`-slice store is exactly what serializes efficiently
  (the Python unified store already spills to `content.blob` + `content.idx` span
  tables); but that lives behind the store trait, added only if needed (YAGNI).

---

## 6. Resolved decisions

1. **Handle width / arena eviction.** Handles are dense `u32` (cap 4B unique
   segments); the contract is whole-dataset-in-memory with no eviction. Revisit only
   for trace files that exceed memory.
2. **Text interning granularity.** V1 interns whole messages/text items — simpler and
   enough for dedup. Alternate granularity (LCP-trie lowering, per
   `2026-06-28-weka-segment-trie-ir-design.md`) remains implementable behind
   `SegmentStore`/the write-side interner without changing conversations.
3. **When to tokenize.** Tokenization happens during composition through local
   tiktoken or Hugging Face tokenizer implementations; token IDs are authoritative for
   message/text identity (so ids are token-keyed and OSL/ISL are known).
4. **`build_body` override splicing.** Raw-object overrides use closing-brace tail
   splicing and never parse or rewrite the authored object; message assembly
   clones/concatenates pre-serialized `Bytes` slices; dynamic graph replies retain
   their encoded wire representation (`graph_segment_unified_store.py:497`).

---

## 7. Status and proof

The dataset/segment seam is realized as the `aiperf_runtime::dataset` module
(`rust/runtime/src/dataset*`) and is shared by `aiperf --execute`, evaluator-authored
static accuracy, Graph-IR (`aiperf_runtime::graph`), and the library-only offline adapter.
There is no mmap/backing-store/client-store or graph-private segment-store fallback.

Proof is executable and self-contained: the dataset suite passes under
`cargo test -p aiperf --lib`, clippy is clean under `-D warnings`, and the native
runtime suite passes under `cargo test -p aiperf --all-targets`. Full-product coverage
lives in the separate `aiperf-e2e-tests` crate (`rust/e2e`), which boots the
`aiperf-mock-server` router on a real loopback port and drives the product
`aiperf profile` frontend against it as a subprocess. In particular,
`rust/e2e/tests/test_chat.rs::test_multi_turn_resends_history` runs the product over a
multi-turn dataset and proves that a live turn's reply is carried into the next
request's resent history. The dataset suite also exercises FFmpeg-backed audio/video
generation and ASR normalization, exact raw replay, all four context modes, loader
auto-detection, fixed/prefix hashing, sampler reproducibility, Hugging Face pagination
and revision pinning, token-native validation, and endpoint-specific request
construction.

Dataset storage, validation, sampling, and materialization remain clock/backend-neutral
for the in-process offline engine sink and authored DAG branch-policy execution, which
are owned by their separate companion specifications.
