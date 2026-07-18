import type { SlideDefinition } from "../../core/types";

type StepInput = Omit<SlideDefinition, "narration">;

const STEPS: readonly StepInput[] = [
  {
    eyebrow: "Overview",
    title: "Rows in → wire bytes out",
    lede:
      "Every request body starts as a dataset row. SegmentPool interns content-addressed segments; freeze drops the hash map into InMemorySegmentStore; BodyPlan plus JsonBodyMaterializer splice handles into Bytes on the hot path.",
    term: {
      word: "serialize once",
      meaning: "Content is serialized at intern time; dispatch clones prebuilt wire bytes instead of re-encoding payloads.",
    },
    points: [
      "`Handle(u32)` is the dense public address; `SegmentId([u8;32])` is the blake3 dedup key.",
      "Freeze discards the id→handle map; handles remain valid in the store.",
      "Evidence: `rust/runtime/src/dataset/segment.rs`, `body_plan.rs`.",
    ],
    caption: "BUILD → FREEZE → DISPATCH · zero content re-serialize on the hot path.",
  },
  {
    eyebrow: "SegmentPool",
    title: "Content-addressed interning",
    lede:
      "SegmentPool is a mutable arena plus HashMap of SegmentId to Handle. A child's identity folds the parent's content hash, so identical prefixes across conversations collapse to the same handle even when load order differs.",
    term: {
      word: "prefix-dependent id",
      meaning: "Child SegmentId hashes parent SegmentId plus domain payload bytes, not insertion index.",
    },
    points: [
      "`Segment { id, parent, payload }` lives in a dense Vec arena.",
      "Dedup hits reuse handles; only novel content allocates.",
      "`freeze()` yields `InMemorySegmentStore` with `Box<[Segment]>`.",
    ],
    caption: "Same parent chain + same bytes → same Handle.",
  },
  {
    eyebrow: "Payloads",
    title: "Six disjoint blake3 domains",
    lede:
      "Payload is one of Message, Text, Raw, TokenIds, Media, or TraceHashIds. Each hashes under its own domain prefix framed by `aiperf-dataset-segment-v1` and the parent id, so identical bytes in different domains never collide.",
    term: {
      word: "SegmentDomain",
      meaning: "Discriminant that drives dispatch formatting: Message array, Raw complete body, or TokenIds native path.",
    },
    points: [
      "Message carries role, wire Bytes, and token ids.",
      "Raw is a complete JSON body or field; TokenIds stay token-native.",
      "TraceHashIds retain simulator cache identity without prompt bytes.",
    ],
    caption: "HASH_VERSION · domain prefix · parent id · payload recipe → SegmentId.",
  },
  {
    eyebrow: "BodyPlan",
    title: "Shape now, splice bytes later",
    lede:
      "BodyPlan declares which JSON fields are literals versus segment handles. JsonBodyMaterializer walks the plan once: literals are the only hot-path serde work; segment slots clone pre-serialized wire bytes from the store.",
    term: {
      word: "JsonBodyMaterializer",
      meaning: "Shared walker that turns a BodyPlan plus SegmentStore into dispatch Bytes.",
    },
    points: [
      "Endpoints format BodyPlans in `rust/runtime/src/endpoints/`.",
      "Dataset precomputes plans for static turns where possible.",
      "Overrides merge stream flags and limits without rewriting message wires.",
    ],
    caption: "literals serialize · segment wires clone · Overrides patch the tail.",
  },
  {
    eyebrow: "Prefix trie",
    title: "Shared prefixes and LCP lowering",
    lede:
      "Composers keep a running parent Handle per conversation so each turn extends a chain. Recorded WEKA and Dynamo traces lower through an LCP trie over block hashes, reusing interned handles for the longest shared prefix.",
    term: {
      word: "rebase",
      meaning: "Re-intern conversation handles under a new system/user context root so blake3 ids reflect the injected prefix.",
    },
    points: [
      "`dataset/compose.rs` owns conversation chains and rebase.",
      "`graph/recorded/trie/` resolves content parents from hash_ids.",
      "Shared system and user turns store once; branches allocate only novel replies.",
    ],
    caption: "One shared prefix · many conversation branches by handle.",
  },
  {
    eyebrow: "Dispatch",
    title: "Turn.body precedence is domain-driven",
    lede:
      "`Turn.dispatch_body` builds one SmallVec of handles: raw payload first, then token ids, else the message list. The domain of the leading handle decides whether materialization takes the complete-body, token-native, or message-array path.",
    term: {
      word: "dispatch_body",
      meaning: "Precedence helper in `dataset/model.rs` that replaces the old multi-field body selection.",
    },
    points: [
      "Raw + TokenIds may coexist as `[raw, token]` for validation backends.",
      "Messages fill the body only when neither raw nor token ids are set.",
      "Graph HTTP can splice message wires directly; scheduled paths use BodyPlan.",
    ],
    caption: "Raw → TokenIds → Messages · domain chooses the materializer path.",
  },
] as const;

const NARRATION = [
  "Dataset rows become wire bytes through SegmentPool, a frozen segment store, and BodyPlan splicing. Content serializes once; the hot path clones handles.",
  "SegmentPool interns by blake3 content identity. Parent hashes fold into children, so shared prefixes collapse to the same dense handle across conversations.",
  "Six payload domains hash under disjoint prefixes. Message, text, raw, token ids, media, and trace hashes never collide even when the bytes look identical.",
  "BodyPlan shapes the JSON object. JsonBodyMaterializer serializes only literals and clones prebuilt segment wires from the store.",
  "Conversation composers and recorded-trace LCP tries keep shared prefixes stored once. Branches allocate only the novel suffix handles.",
  "Turn.body is one precedence vector. Raw leads, token ids follow, otherwise messages fill the body, and SegmentDomain selects the materializer path.",
] as const;

export const SLIDES: readonly SlideDefinition[] = STEPS.map((slide, index) => ({
  ...slide,
  narration: NARRATION[index] ?? "",
}));
