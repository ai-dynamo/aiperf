/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the Python Graph Workload plane: Dynamo trace ingest,
//! workload construction, and the asyncio dataflow executor.
//!
//! Grounded in `src/aiperf/dataset/graph/` and `src/aiperf/graph/` on the
//! `ajc/dynamo-graph-ir` worktree; captions carry file and symbol.

import type { DeckDefinition, SlideDefinition } from "../../deck/types.js";
import { band, card, chip, fanIn, fanOut, link, note, timeline } from "../shared/diagram.js";

const SLIDES: readonly SlideDefinition[] = [
  {
    id: "planes",
    eyebrow: "01 · THE SHAPE OF THE SYSTEM",
    title: "Two planes, and only one of them parses",
    lede:
      "The DatasetManager is the sole parser. It writes an mmap store and a content-free sidecar; the TimingManager never sees prompt text at all.",
    narration:
      "Start with the split that everything else depends on. Only one process parses a workload: the dataset manager. It builds a memory-mapped segment store and a sidecar with every prompt stripped out. The timing manager reads only that sidecar, so it never touches content. The worker reads envelopes back out of the store. What holds the three together is a single shared number — the node ordinal, derived from that one parse.",
    caption:
      "dataset/graph/workload_detect.py:16-25 states the two-plane split; store_build.py GraphStoreBuilder; graph_meta_sidecar.py:76 strip_replay_text empties prompts but keeps the `trie` key so routing still works.",
    nodes: [
      band("b-p", "PARSE ONCE", { col: 0, row: 1 }),
      card("ds", "DatasetManager", "the only parser", "control", { col: 0, row: 1 }),
      card("store", "mmap segment store", "content.blob + nodes.blob", "data", { col: 1.3, row: 0 }),
      card("side", "graph_meta sidecar", "prompts stripped", "muted", { col: 1.3, row: 2 }),
      card("tm", "TimingManager", "never parses", "time", { col: 2.6, row: 2 }),
      card("wk", "Worker", "reads envelopes", "done", { col: 2.6, row: 0 }),
      chip("ord", "node ordinal", { col: 1.3, row: 1 }),
      note(
        "inv",
        "One number ties the planes together",
        "Build-time ordinals and dispatch-time catalog ordinals both come from the same parse. That is what makes the worker read the right envelope.",
        { col: 3.7, row: 1 },
      ),
    ],
    edges: [
      ...fanOut("ds", ["store", "side"], "data"),
      link("store", "wk", "data"),
      link("side", "tm", "time"),
      link("ds", "ord", "control", "slow"),
    ],
    revealOrder: ["b-p", "ds", "store", "side", "wk", "tm", "ord", "inv"],
  },
  {
    id: "forest",
    eyebrow: "02 · DYNAMO TRACE",
    title: "A trace is a forest, not a list",
    lede:
      "Sessions link by parent_trajectory_id into trees. Each tree lowers independently, so cross-tree edges vanish by construction.",
    narration:
      "A Dynamo trace arrives as a flat stream of request-end and tool events. The reader folds them into sessions, and sessions link to their parents to form a forest. That forest is partitioned into trees, and here is the useful part: each tree is lowered completely independently, which means cross-tree edges cannot exist by construction rather than by a check. The trees are then merged into one multi-graph workload keyed by reference.",
    caption:
      "adapters/dynamo/trace_reader.py:667 iter_session_records folds records; trace.py:513 group_chains_into_trees and :544 root_of_sessions; :484 _finalize_parsed_graph then merge_parsed_graphs.",
    nodes: [
      band("b-r", "FLAT RECORD STREAM", { col: 0, row: 1.2 }),
      card("rec", "request_end / tool_*", "one JSONL line each", "muted", { col: 0, row: 1.2 }),
      chip("fold", "fold by session", { col: 1.1, row: 1.2 }),

      band("b-f", "SESSION FOREST", { col: 2, row: -0.4 }),
      card("r1", "root session A", "trajectory root", "control", { col: 2, row: -0.4 }),
      card("c1", "child A1", "parent_trajectory_id", "data", { col: 3.1, row: -0.9 }),
      card("c2", "child A2", "parent_trajectory_id", "data", { col: 3.1, row: 0.1 }),
      card("r2", "root session B", "separate tree", "control", { col: 2, row: 1.6 }),
      card("c3", "child B1", "parent_trajectory_id", "data", { col: 3.1, row: 1.6 }),

      card("merge", "multi-graph workload", "graphs[root_id]", "done", { col: 2.55, row: 2.9 }),
      note(
        "iso",
        "Isolation by construction",
        "Each tree lowers on its own, so a cross-tree edge is not rejected — it is unrepresentable.",
        { col: 4.3, row: 0.4 },
      ),
    ],
    edges: [
      link("rec", "fold", "muted"),
      link("fold", "r1", "control", "slow"),
      link("fold", "r2", "control", "slow"),
      ...fanOut("r1", ["c1", "c2"], "data"),
      link("r2", "c3", "data"),
      ...fanIn(["r1", "r2"], "merge", "done", "slow"),
    ],
    revealOrder: ["b-r", "rec", "fold", "b-f", "r1", "c1", "c2", "r2", "c3", "merge", "iso"],
  },
  {
    id: "trie",
    eyebrow: "03 · BUILDING THE WORKLOAD",
    title: "Four stages, with a shortcut through the middle",
    lede:
      "Content parents, then timing warp and edges, then block tags, then per-node emission — which splices the parent's prefix instead of re-decoding it.",
    narration:
      "The workload build runs in four stages. First, resolve which earlier request each one shares a prefix with. Second, warp idle time and derive the timing edges. Third, freeze the block tags. Fourth, emit each node. The fourth stage has a shortcut worth knowing: rather than re-decoding a shared prefix, it reuses the parent's segment-id chain and emits only the fragment that straddles the boundary plus the genuinely new region.",
    caption:
      "segment_ir/trie_content.py:10-38 documents the four stages; :1048 build_trie_ir; prefix splice via bisect_right over msg_end_blocks reusing _EmissionRecord:538; assert_covered_isl:996 gates the assembled token count.",
    nodes: [
      band("b-s", "FOUR STAGES", { col: 0, row: 0.4 }),
      card("s1", "content parents", "prefix automaton", "data", { col: 0, row: 0.4 }),
      card("s2", "warp + edges", "timing", "time", { col: 1.15, row: 0.4 }),
      card("s3", "block tags", "role, new-message", "muted", { col: 2.3, row: 0.4 }),
      card("s4", "emit nodes", "segments per node", "control", { col: 3.45, row: 0.4 }),

      card("splice", "prefix splice", "reuse parent sid chain", "done", { col: 2.3, row: 1.7 }),
      chip("gate", "ISL gate", { col: 3.6, row: 1.7 }),
      note(
        "why",
        "The shortcut is the optimisation",
        "A shared block is decoded once. The child re-emits only the straddling fragment and its new region, and the assembled token count is asserted against the recorded ISL.",
        { col: 0, row: 2.6 },
      ),
    ],
    edges: [
      link("s1", "s2", "data"),
      link("s2", "s3", "time"),
      link("s3", "s4", "muted"),
      link("s1", "splice", "done", "slow"),
      link("splice", "s4", "done"),
      link("splice", "gate", "control"),
    ],
    revealOrder: ["b-s", "s1", "s2", "s3", "s4", "splice", "gate", "why"],
  },
  {
    id: "automaton",
    eyebrow: "04 · TRIE GENERATION",
    title: "One pass, two annotations per state",
    lede:
      "A flat-int-state prefix automaton. Each state records a terminal owner, overwritten always, and a passer owner, set once — opposite tie-breaks on purpose.",
    narration:
      "Finding each request's content parent could be a quadratic scan over every earlier request. Instead it is one linear pass through a prefix automaton built from flat integer states. Every state carries two annotations. The terminal owner is overwritten every time, so a full-prefix tie goes to the most recent node — a genuine continuation, nearest in time and most likely still cached. The passer owner is set once, so a partial-overlap tie goes to the earliest node, the one that established the branch point. Opposite rules, and both deliberate.",
    caption:
      "trie_content.py:558 resolve_content_parents; the walk at :613-630 latches best_full and best_partial independently. terminal overwrites at :700, passer is set-once at :698. Complexity claim at :569-576 — linear in total hash length, against a naive O(n squared times m).",
    nodes: [
      band("b-a", "ONE AUTOMATON", { col: 0, row: 0.9 }),
      card("s0", "state 0", "root", "muted", { col: 0, row: 0.9 }),
      card("s1", "state 1", "hash a", "control", { col: 1.1, row: 0.9 }),
      card("s2", "state 2", "hash b", "control", { col: 2.2, row: 0.3 }),
      card("s3", "state 3", "hash c", "control", { col: 2.2, row: 1.6 }),

      band("b-t", "TWO ANNOTATIONS PER STATE", { col: 3.4, row: 0 }),
      card("term", "terminal", "overwrite always - most recent wins", "done", { col: 3.4, row: 0 }),
      card("pass", "passer", "set once - earliest wins", "time", { col: 3.4, row: 1.1 }),

      card("full", "best_full", "a continuation", "done", { col: 3.4, row: 2.4 }),
      card("part", "best_partial", "a branch point", "time", { col: 3.4, row: 3.4 }),
      note(
        "why",
        "Why the tie-breaks differ",
        "Most-recent for a continuation keeps the parent nearest in time. Earliest for a branch point keeps every sibling of a fork on the same stable ancestor instead of on whichever sibling ran last.",
        { col: 0, row: 2.7 },
      ),
    ],
    edges: [
      link("s0", "s1", "control"),
      ...fanOut("s1", ["s2", "s3"], "control"),
      link("term", "full", "done"),
      link("pass", "part", "time"),
    ],
    revealOrder: ["b-a", "s0", "s1", "s2", "s3", "b-t", "term", "pass", "full", "part", "why"],
  },
  {
    id: "geometry",
    eyebrow: "05 · TRIE GENERATION",
    title: "Whole blocks only, and the tail is popped",
    lede:
      "Every frozen field is a whole-block count on the block-aligned grid. The partial tail is excluded, and a trailing partial hash is dropped outright.",
    narration:
      "Block geometry is the single source every later stage reads. The key rule: all of it is whole blocks. The partial tail — the remainder of tokens past the last full block — is deliberately excluded from every frozen field and synthesized separately at assembly time. Upstream, a trailing partial hash is popped entirely, because engines cache and share full blocks only, so a partial block can never be a prefix-cache hit. And a turn with no recorded hashes gets a chain of virtual negative ids, so consecutive turns of a session still share a prefix.",
    caption:
      "trie_content.py:101-126 compute_turn_block_geometry, 'the SINGLE geometry source'; the partial-tail exclusion at :87-92. The pop is trie_lowering.py:172-175 with three reasons at :145-151. Virtual chain at :179-190; trace_reader.py:135-148 rejects negative recorded hashes because they collide with that namespace.",
    nodes: [
      band("b-g", "THE BLOCK GRID", { col: 0, row: 0.5 }),
      card("b1", "block 0", "full - hashed", "done", { col: 0, row: 0.5 }),
      card("b2", "block 1", "full - hashed", "done", { col: 1.05, row: 0.5 }),
      card("b3", "block 2", "full - hashed", "done", { col: 2.1, row: 0.5 }),
      card("tail", "partial tail", "popped - never a cache hit", "failure", { col: 3.15, row: 0.5 }),

      band("b-f", "FROZEN FIELDS", { col: 0, row: 1.9 }),
      card("f1", "m_covered", "whole blocks", "data", { col: 0, row: 1.9 }),
      card("f2", "lcp", "whole blocks", "data", { col: 1.05, row: 1.9 }),
      card("f3", "new_blocks_count", "whole blocks", "data", { col: 2.1, row: 1.9 }),
      chip("synth", "tail synthesized later", { col: 3.15, row: 1.9 }),

      note(
        "virt",
        "No recorded hashes? Virtual ids",
        "A turn with no replay gets negative ids counting down per session, so consecutive turns still share a prefix. Recorded negatives are rejected — they would collide with that namespace.",
        { col: 0, row: 2.9 },
      ),
    ],
    edges: [
      link("b1", "b2", "done"),
      link("b2", "b3", "done"),
      link("b3", "tail", "failure"),
      link("tail", "synth", "time", "slow"),
      ...fanIn(["f1", "f2", "f3"], "synth", "data", "slow"),
    ],
    revealOrder: ["b-g", "b1", "b2", "b3", "tail", "b-f", "f1", "f2", "f3", "synth", "virt"],
  },
  {
    id: "tags",
    eyebrow: "06 · TRIE GENERATION",
    title: "Tags freeze at first materialization",
    lede:
      "A block tag is a role plus a starts-new-message flag, assigned by the node that first covers that block and then never re-tagged.",
    narration:
      "Every covered block gets a tag: a role, and whether it opens a new message. The node that first materializes a block position assigns that tag, and it is then frozen forever. This is what makes prefix sharing real. Two requests sharing a block prefix read the identical tags on it, which produces identical message grouping, which produces the identical segment id chain — and only then is it genuinely the same cache prefix. Re-tagging a shared block on a later turn would silently break that identity, which is exactly why the tags freeze.",
    caption:
      "trie_content.py:757 _assign_block_tags_and_inheritance; the freezing rationale at :829-836. The trailing-user flip is block_role_split:177-180 — a node whose new region is all assistant flips its own last block to user. compute_asst_caps:706 caps an ancestor for the degenerate pull-back case.",
    nodes: [
      band("b-p", "PARENT · TAGS FROZEN", { col: 0, row: 0.4 }),
      card("p1", "block 0", "user", "data", { col: 0, row: 0.4 }),
      card("p2", "block 1", "assistant", "control", { col: 1.05, row: 0.4 }),
      card("p3", "block 2", "user - flipped", "time", { col: 2.1, row: 0.4 }),

      band("b-c", "CHILD · INHERITS VERBATIM", { col: 0, row: 1.8 }),
      card("c1", "block 0", "same tag, copied", "data", { col: 0, row: 1.8 }),
      card("c2", "block 1", "same tag, copied", "control", { col: 1.05, row: 1.8 }),
      card("c3", "new region", "its own tags", "done", { col: 2.1, row: 1.8 }),

      chip("same", "same sid chain", { col: 3.2, row: 1.1 }),
      note(
        "flip",
        "The trailing-user flip",
        "A node whose whole new region is assistant flips its own last new block to user, so a turn boundary always lands on a user block and framing stays cache-safe.",
        { col: 0, row: 2.9 },
      ),
    ],
    edges: [
      link("p1", "c1", "data"),
      link("p2", "c2", "control"),
      link("p3", "c3", "time", "slow"),
      ...fanIn(["c1", "c2"], "same", "done", "slow"),
    ],
    revealOrder: ["b-p", "p1", "p2", "p3", "b-c", "c1", "c2", "c3", "same", "flip"],
  },
  {
    id: "splice",
    eyebrow: "07 · TRIE GENERATION",
    title: "Reuse every whole message inside the boundary",
    lede:
      "A bisect over the parent's message end-blocks splits reused from fresh. Only the straddling message and the new region are decoded and hashed again.",
    narration:
      "Here is where the build avoids going quadratic. The child knows how many blocks it inherits, so a bisect over the parent's message end-blocks finds how many whole parent messages fit entirely inside that boundary. Those segment ids are spliced in directly — not re-decoded, not re-hashed. Only the message straddling the boundary, plus the genuinely new region, are emitted fresh, chained onto the spliced tip. And because the child's inherited tags are the parent's copied verbatim, a re-emitted prefix would have produced the identical ids anyway, so the output is byte-identical.",
    caption:
      "trie_content.py:1134 bisect_right over rec.msg_end_blocks; _EmissionRecord:537 carries msg_end_blocks, cum_token_counts and the shared prompt_path list. The reuse boundary must come from geometric inherited, not the tags — :1117-1120 warns tags can coincide past the lcp while hash ids differ.",
    nodes: [
      band("b-p", "PARENT PROMPT PATH", { col: 0, row: 0.4 }),
      card("m1", "message 0", "sid reused", "done", { col: 0, row: 0.4 }),
      card("m2", "message 1", "sid reused", "done", { col: 1.05, row: 0.4 }),
      card("m3", "message 2", "straddles boundary", "time", { col: 2.1, row: 0.4 }),

      chip("bisect", "bisect_right(inherited)", { col: 1.6, row: 1.4 }),

      band("b-c", "CHILD PROMPT PATH", { col: 0, row: 2.4 }),
      card("s1", "spliced", "zero decode", "done", { col: 0, row: 2.4 }),
      card("f1", "fresh fragment", "decoded + hashed", "time", { col: 1.15, row: 2.4 }),
      card("f2", "new region", "decoded + hashed", "data", { col: 2.3, row: 2.4 }),

      note(
        "eq",
        "Byte-identical either way",
        "The inherited tags are the parent's copied verbatim, so re-emitting the prefix would derive the same ids and a spliced add would have been a dedup no-op. The splice only removes the quadratic re-decode.",
        { col: 3.4, row: 1.4 },
      ),
    ],
    edges: [
      ...fanIn(["m1", "m2", "m3"], "bisect", "muted"),
      link("bisect", "s1", "done"),
      link("bisect", "f1", "time"),
      link("s1", "f1", "done", "slow"),
      link("f1", "f2", "data"),
    ],
    revealOrder: ["b-p", "m1", "m2", "m3", "bisect", "b-c", "s1", "f1", "f2", "eq"],
  },
  {
    id: "determinism",
    eyebrow: "08 · TRIE GENERATION",
    title: "The gate counts what was actually assembled",
    lede:
      "assert_covered_isl checks reused plus fresh tokens against the covered-block target — not a value re-derived from the tags, which would be tautological.",
    narration:
      "Two guards keep the build honest. The ISL gate compares the tokens actually assembled — reused plus fresh — against the covered-block target. Deliberately not a count re-derived from the tags, because that would be tautological and prove nothing. And the target is the covered count, not blocks-times-size, because a legitimate recording may store fewer hash blocks than its declared input length would imply. Underneath sits a determinism contract: the same hash id must decode to the same tokens for every call in one build. The reuse path decodes each shared block exactly once, so a drifting callback would no longer be re-caught downstream.",
    caption:
      "trie_content.py:996 assert_covered_isl, target is min(len(hash_ids), input_length // bs) * bs with the rationale at :1001-1006; called at :1166 with reused plus fresh. ReconCallbacks determinism contract at :298-311. Module header :8 — 'Any behavior change here breaks recorded-trace byte-exactness'.",
    nodes: [
      band("b-l", "TOKEN LEDGER", { col: 0, row: 0.6 }),
      card("re", "reused tokens", "from the parent record", "done", { col: 0, row: 0.2 }),
      card("fr", "fresh tokens", "decoded this node", "time", { col: 0, row: 1.2 }),
      chip("sum", "sum", { col: 1.2, row: 0.7 }),
      card("gate", "covered-block target", "min(hashes, in // bs) * bs", "control", { col: 1.9, row: 0.7 }),
      card("ok", "build proceeds", "or TrieISLMismatchError", "data", { col: 3.1, row: 0.7 }),

      band("b-d", "DETERMINISM CONTRACT", { col: 0, row: 2.4 }),
      card("d1", "same hash id", "same tokens, every call", "muted", { col: 0, row: 2.4 }),
      card("d2", "shared block decoded once", "reuse path", "done", { col: 1.3, row: 2.4 }),
      note(
        "drift",
        "A drifting callback is not re-caught",
        "Because the shared prefix is decoded once and spliced, a callback that returned different tokens at the child would no longer trip the ISL gate there.",
        { col: 2.6, row: 2.4 },
      ),
    ],
    edges: [
      ...fanIn(["re", "fr"], "sum", "done"),
      link("sum", "gate", "control"),
      link("gate", "ok", "data"),
      link("d1", "d2", "muted"),
    ],
    revealOrder: ["b-l", "re", "fr", "sum", "gate", "ok", "b-d", "d1", "d2", "drift"],
  },
  {
    id: "parents",
    eyebrow: "09 · CONTENT vs TIMING",
    title: "The content parent is not the timing parent",
    lede:
      "Prefix sharing can reach arbitrarily far back. Using that same link for the firing delay would accumulate the whole distance — the aggregate-timestamp bug.",
    narration:
      "These two parent relationships look alike and must never be conflated. The content parent answers: whose prompt prefix do I share? That branch point can be arbitrarily far back in the trace. The timing parent answers: what do I fire after? If you route the firing delay through the content link, the delay becomes the cumulative warped distance back to that ancestor. The code calls this out by name — the aggregate-timestamp bug.",
    caption:
      "trie_content.py:365-373 TrieNode.content_parent — 'CONTENT/PROMPT ONLY'. Timing edges come from interval_order.py:49 build_interval_edges, a separate derivation.",
    nodes: [
      band("b-c", "CONTENT PARENT", { col: 0, row: 0.2 }),
      card("anc", "request 3", "shared prefix owner", "data", { col: 0, row: 0.2 }),
      card("mid1", "request 40", "unrelated", "muted", { col: 1.2, row: 0.2 }),
      card("me1", "request 91", "splices from 3", "data", { col: 2.4, row: 0.2 }),

      band("b-t", "TIMING PARENT", { col: 0, row: 1.9 }),
      card("prev", "request 90", "finished just before", "time", { col: 1.2, row: 1.9 }),
      card("me2", "request 91", "fires after 90", "time", { col: 2.4, row: 1.9 }),

      note(
        "bug",
        "The aggregate-timestamp bug",
        "Route the delay through the content link and it becomes the cumulative warped distance back to an arbitrarily old ancestor.",
        { col: 3.6, row: 1 },
      ),
    ],
    edges: [
      link("anc", "me1", "data", "slow"),
      link("prev", "me2", "time"),
      link("mid1", "me1", "muted", "slow"),
    ],
    revealOrder: ["b-c", "anc", "mid1", "me1", "b-t", "prev", "me2", "bug"],
  },
  {
    id: "warp",
    eyebrow: "10 · IDLE WARP",
    title: "Cap the dead air, never the request",
    lede:
      "The warp collapses gaps between the running-max end and the next start. Capping start-to-start would eat into a long request's own service time.",
    narration:
      "Recorded traces contain long idle stretches, and replaying them verbatim would take as long as the original session. So gaps get capped. The subtlety is which gap. Capping start-to-start distance eats into a long request's own service time and manufactures overlaps that never happened. The warp instead caps only true dead air — from the running maximum end to the next start. The invariant that falls out is that a warped request keeps its exact service time.",
    caption:
      "trie_content.py:193-212 ActiveIdleWarp docstring; invariant warped_end == warped_start + api_time always holds. TrieNode.end:405 adds api_time raw to the warped start.",
    nodes: [
      band("b-n", "NAIVE · START TO START", { col: 0, row: 0.2 }),
      card("n1", "request X", "long service time", "muted", { col: 0, row: 0.2 }),
      card("n2", "capped gap", "eats into X", "failure", { col: 1.15, row: 0.2 }),
      card("n3", "false overlap", "never happened", "failure", { col: 2.3, row: 0.2 }),

      band("b-w", "WARP · ACTIVE INTERVAL", { col: 0, row: 1.8 }),
      card("w1", "request X", "service time intact", "done", { col: 0, row: 1.8 }),
      card("w2", "dead air only", "running-max end -> next start", "time", { col: 1.15, row: 1.8 }),
      card("w3", "next request", "ordering preserved", "done", { col: 2.3, row: 1.8 }),

      note(
        "inv",
        "The invariant",
        "warped_end equals warped_start plus api_time, always. Service time is never compressed.",
        { col: 3.5, row: 1 },
      ),
    ],
    edges: [
      link("n1", "n2", "failure"),
      link("n2", "n3", "failure"),
      link("w1", "w2", "time"),
      link("w2", "w3", "done"),
    ],
    revealOrder: ["b-n", "n1", "n2", "n3", "b-w", "w1", "w2", "w3", "inv"],
  },
  {
    id: "warp-timeline",
    eyebrow: "11 · IDLE WARP",
    title: "The same trace, on both clocks",
    lede:
      "Two 87-second and 10-second dead-air stretches collapse to the 5-second cap. Every bar keeps its width, and sub-A and sub-B still overlap exactly as recorded.",
    narration:
      "Here is the warp applied to a real shape rather than described in the abstract. The top block is the recorded clock: one request, then eighty-seven seconds of nothing, then a burst where two subagents run at the same time. The bottom block is what the runtime replays. The two dead-air stretches have collapsed to the five-second cap, so the trace finishes in thirty-five seconds instead of a hundred and twenty-two. Look at what did not change. Every bar is exactly as wide as it was — service time is never compressed. And sub-A and sub-B still overlap, by the same amount, because the warp shifts whole intervals rather than rescaling the axis.",
    caption:
      "trie_content.py:193-212 ActiveIdleWarp; cuts built by the running-max-end sweep with cap=5s. Bars are that algorithm's output, not illustrative: m1 90->8, a0 94->12, m2 120->33.",
    nodes: [
      timeline(
        "tl",
        {
          title: "Overlapping subagents past a long idle · cap 5s",
          lanes: ["main", "sub-A", "sub-B"],
          bars: [
            { id: "m0", lane: "main", rawStart: 0, rawEnd: 3, warpStart: 0, warpEnd: 3 },
            { id: "m1", lane: "main", rawStart: 90, rawEnd: 93, warpStart: 8, warpEnd: 11 },
            { id: "a0", lane: "sub-A", rawStart: 94, rawEnd: 102, warpStart: 12, warpEnd: 20 },
            { id: "b0", lane: "sub-B", rawStart: 96, rawEnd: 106, warpStart: 14, warpEnd: 24 },
            { id: "a1", lane: "sub-A", rawStart: 102, rawEnd: 107, warpStart: 20, warpEnd: 25 },
            { id: "b1", lane: "sub-B", rawStart: 106, rawEnd: 110, warpStart: 24, warpEnd: 28 },
            { id: "m2", lane: "main", rawStart: 120, rawEnd: 122, warpStart: 33, warpEnd: 35 },
          ],
          gaps: [
            { start: 3, end: 90, idle: 87, capped: true },
            { start: 93, end: 94, idle: 1, capped: false },
            { start: 110, end: 120, idle: 10, capped: true },
          ],
        },
        { col: 0, row: 0 },
      ),
      note(
        "keep",
        "What the warp preserves",
        "Bar widths (service time) and the overlap between sub-A and sub-B. Whole intervals shift left by a cumulative amount; the axis is never rescaled.",
        { col: 0, row: 3.1 },
      ),
      note(
        "cut",
        "What the warp removes",
        "Only true dead air, measured running-max-end to next start. The 1-second gap sits under the cap and is left alone; the 87 and 10 second gaps are cut to 5.",
        { col: 1.1, row: 3.1 },
      ),
    ],
    edges: [],
    revealOrder: ["tl", "keep", "cut"],
  },
  {
    id: "clocks",
    eyebrow: "12 · INTERVAL ORDER",
    title: "Two clocks, two jobs",
    lede:
      "raw_start and raw_end decide who depends on whom. The warped start and end decide how long to wait. Every rule reads one for shape and the other for numbers.",
    narration:
      "Edge derivation runs on two clocks at once, and keeping them straight is the whole trick. The raw interval is what was actually recorded, and it decides structure — who finished before whom. The warped interval is the clock the runtime replays, and it decides delay. Every rule that follows reads raw for the shape of the graph, and warped for the numbers written on it. Conflating them is precisely the bug this module is built to avoid.",
    caption:
      "interval_order.py:5-15 documents the duck-typed surface: raw_start/raw_end are 'the RAW recorded who-finished-before-whom interval'; start/end are 'the idle-gap-warped clock the runtime replays'.",
    nodes: [
      band("b-r", "RAW CLOCK · DECIDES STRUCTURE", { col: 0, row: 0.2 }),
      card("raw", "raw_start / raw_end", "as recorded", "data", { col: 0, row: 0.2 }),
      card("cand", "candidate rule", "finished-before", "data", { col: 1.25, row: -0.3 }),
      card("front", "frontier", "who covers whom", "data", { col: 1.25, row: 0.7 }),

      band("b-w", "WARPED CLOCK · DECIDES DELAY", { col: 0, row: 2 }),
      card("warp", "start / end", "what the runtime replays", "time", { col: 0, row: 2 }),
      card("bind", "binding delay", "end-to-start gap", "time", { col: 1.25, row: 1.5 }),
      card("root", "START offset", "min_start_delay_us", "time", { col: 1.25, row: 2.5 }),

      note(
        "mix",
        "Never cross the streams",
        "Structure from raw, timing from warped. Deriving structure on the warped clock would let idle-gap capping invent and destroy dependencies.",
        { col: 2.5, row: 1.1 },
      ),
    ],
    edges: [
      ...fanOut("raw", ["cand", "front"], "data"),
      ...fanOut("warp", ["bind", "root"], "time"),
    ],
    revealOrder: ["b-r", "raw", "cand", "front", "b-w", "warp", "bind", "root", "mix"],
  },
  {
    id: "candidates",
    eyebrow: "13 · INTERVAL ORDER",
    title: "Three conjuncts, then async exclusion",
    lede:
      "A is a candidate for B when A ranks earlier, A finished before B started on the raw clock, and A is not inside a fire-and-forget subtree that B sits outside of.",
    narration:
      "A candidate predecessor has to satisfy three things at once. It must rank earlier. It must have finished before the target started, on the raw clock. And it must survive async exclusion. That last rule says a fire-and-forget subtree never and-joins the scope that launched it: if a candidate sits under an async boundary the target does not share, it is dropped outright. Note where this happens — exclusion runs before the frontier filter, not after it.",
    caption:
      "interval_order.py:79-86 the three conjuncts; :38 _excluded_async is `not cand.async_ancestors <= target.async_ancestors`. Rank comes from :26 compute_ranks, a linear extension so the rule stays a strict partial order and its reduction is always a DAG.",
    nodes: [
      band("b-p", "EVERY EARLIER NODE", { col: 0, row: 0.8 }),
      card("pool", "all nodes", "the starting pool", "muted", { col: 0, row: 0.8 }),

      card("g1", "rank earlier", "conjunct 1", "control", { col: 1.2, row: -0.3 }),
      card("g2", "raw_end <= raw_start", "conjunct 2", "data", { col: 1.2, row: 0.8 }),
      card("g3", "not async-excluded", "conjunct 3", "time", { col: 1.2, row: 1.9 }),

      chip("and", "all three", { col: 2.45, row: 0.8 }),
      card("cands", "candidate set", "before reduction", "done", { col: 3.05, row: 0.8 }),
      card("drop", "fire-and-forget child", "dropped here", "failure", { col: 3.05, row: 2.2 }),

      note(
        "why",
        "Why rank at all",
        "The finished-before test alone could admit a cycle if the warp ever reordered. Requiring rank(A) < rank(B) keeps the relation a strict partial order.",
        { col: 0, row: 2.6 },
      ),
    ],
    edges: [
      ...fanOut("pool", ["g1", "g2", "g3"], "muted"),
      ...fanIn(["g1", "g2", "g3"], "and", "control"),
      link("and", "cands", "done"),
      link("g3", "drop", "failure", "slow"),
    ],
    revealOrder: ["b-p", "pool", "g1", "g2", "g3", "and", "cands", "drop", "why"],
  },
  {
    id: "frontier",
    eyebrow: "14 · INTERVAL ORDER",
    title: "Keep only the maximal predecessors",
    lede:
      "Drop c when a later candidate d covers it — but only when the covering edge c to d actually exists.",
    narration:
      "The candidate set is far too large, so it is transitively reduced to a frontier: keep only the maximal finished-before candidates, dropping any that a later one already covers. The subtlety is in what counts as covering. The covering edge has to actually exist, which means the later candidate must not itself async-exclude the one being dropped. Without that check a main-chain node outside the async subtree would drop a candidate while carrying no edge to it, and the recorded ordering inside that subtree would silently vanish.",
    caption:
      "interval_order.py:100-107; the guard is `not _excluded_async(c, d)` at :104, with the rationale at :62-64. The scan is `cands[i + 1:]`, sound only because cands was built from the rank-sorted by_rank at :76.",
    nodes: [
      band("b-c", "CANDIDATES", { col: 0, row: 0.2 }),
      card("c1", "c1", "covered by c3", "muted", { col: 0, row: -0.6 }),
      card("c2", "c2", "covered by c3", "muted", { col: 0, row: 0.4 }),
      card("c3", "c3", "maximal", "done", { col: 0, row: 1.4 }),
      card("c4", "c4 (async subtree)", "d carries no edge to it", "failure", { col: 0, row: 2.4 }),

      chip("red", "transitive reduction", { col: 1.3, row: 0.9 }),

      band("b-f", "FRONTIER", { col: 2.5, row: 0.9 }),
      card("f1", "c3", "kept", "done", { col: 2.5, row: 0.9 }),
      card("f2", "c4", "kept - not truly covered", "time", { col: 2.5, row: 2.1 }),

      note(
        "sub",
        "Covered is not enough",
        "A later candidate only covers c if it would itself carry the c edge. A main-chain node outside c's async subtree cannot, so c survives the filter.",
        { col: 3.7, row: 1.2 },
      ),
    ],
    edges: [
      ...fanIn(["c1", "c2", "c3"], "red", "muted"),
      link("c4", "red", "failure", "slow"),
      link("red", "f1", "done"),
      link("red", "f2", "time"),
    ],
    revealOrder: ["b-c", "c1", "c2", "c3", "c4", "red", "b-f", "f1", "f2", "sub"],
  },
  {
    id: "binding",
    eyebrow: "15 · INTERVAL ORDER",
    title: "One edge carries the delay, the rest wait at zero",
    lede:
      "The latest-ending frontier member carries the warped gap. Every other frontier edge is an AND-join at delay zero. An empty frontier roots the node at START instead.",
    narration:
      "Now the delays. Exactly one frontier member binds — the one that ends latest on the warped clock — and it carries the entire end-to-start gap, clamped at zero. Every other frontier edge is emitted with a delay of zero. Those are not decorative: the node genuinely waits on all of them, they simply contribute no time. And if the frontier came back empty, the node has no predecessor at all, so it roots at start with a minimum start delay taken from its own warped arrival.",
    caption:
      "interval_order.py:108 `binding = max(frontier, key=lambda c: c.end)`; :113-117 the delay is `max(0.0, node.start - c.end)` on the warped clock for the binding edge and 0.0 for every other; :87-95 the empty-frontier case uses min_start_delay_us from START.",
    nodes: [
      band("b-f", "FRONTIER", { col: 0, row: 0.3 }),
      card("b1", "ends latest", "BINDING", "done", { col: 0, row: 0.3 }),
      card("b2", "frontier sibling", "AND-join", "control", { col: 0, row: 1.3 }),
      card("b3", "frontier sibling", "AND-join", "control", { col: 0, row: 2.3 }),

      chip("d1", "warped gap", { col: 1.25, row: 0.3 }),
      chip("d2", "delay 0.0", { col: 1.25, row: 1.3 }),
      chip("d3", "delay 0.0", { col: 1.25, row: 2.3 }),
      card("node", "the node", "waits on all three", "data", { col: 2.1, row: 1.3 }),

      band("b-e", "EMPTY FRONTIER", { col: 0, row: 3.5 }),
      card("start", "START", "min_start_delay_us", "time", { col: 0, row: 3.5 }),
      card("orphan", "no predecessor", "roots at its own arrival", "time", { col: 1.4, row: 3.5 }),

      note(
        "zero",
        "Zero-delay edges still gate",
        "They contribute no time but the node genuinely waits on them, so a slow sibling still holds the fan-in.",
        { col: 3.2, row: 1 },
      ),
    ],
    edges: [
      link("b1", "d1", "done"),
      link("b2", "d2", "control"),
      link("b3", "d3", "control"),
      ...fanIn(["d1", "d2", "d3"], "node", "data"),
      link("start", "orphan", "time"),
    ],
    revealOrder: ["b-f", "b1", "b2", "b3", "d1", "d2", "d3", "node", "b-e", "start", "orphan", "zero"],
  },
  {
    id: "anchors",
    eyebrow: "16 · INTERVAL ORDER",
    title: "A mid-flight child replaces its whole edge set",
    lede:
      "When a child's recorded start falls inside its causal parent's interval, every interval-order edge is discarded and replaced by one start-anchored edge.",
    narration:
      "Finally, an override that discards everything derived so far. If a node names a causal parent in the set, and its recorded start falls strictly inside that parent's recorded interval, then the child was running mid-flight. Its entire incoming edge set is replaced by a single edge anchored to the parent's dispatch, carrying the start-to-start gap. The runtime then schedules it when the parent dispatches rather than when it finishes. And if the parent streamed, and the child began at or after the first token, the edge carries a first-token offset too, so the runtime can re-anchor onto the token it actually observes and fall back to dispatch if none arrives.",
    caption:
      "interval_order.py:158 the containment test `parent.raw_start <= node.raw_start < parent.raw_end`; :165 assigns rather than appends, so the interval-order edges are discarded; :161-164 adds delay_after_predecessor_first_token_us only when the parent streamed and the child started at or after ttft.",
    nodes: [
      band("b-t", "THE TEST", { col: 0, row: 0.6 }),
      card("test", "started mid-flight?", "parent.raw_start <= start < parent.raw_end", "control", { col: 0, row: 0.6 }),

      card("no", "parent already finished", "keeps interval-order edges", "muted", { col: 1.35, row: -0.3 }),
      card("yes", "replace all in-edges", "one start-anchored edge", "done", { col: 1.35, row: 1.5 }),

      band("b-a", "WHICH ANCHOR", { col: 2.7, row: 0.9 }),
      card("a1", "dispatch anchor", "start-to-start gap", "time", { col: 2.7, row: 0.9 }),
      card("a2", "plus first-token", "only if parent streamed", "data", { col: 2.7, row: 2 }),

      note(
        "why",
        "Causal, not wall-clock",
        "Recorded mid-flight concurrency - subagent spawns, aux tool calls - tracks the parent causally instead of freezing to the recorded wall clock.",
        { col: 0, row: 2.8 },
      ),
    ],
    edges: [
      link("test", "no", "muted"),
      link("test", "yes", "done"),
      link("yes", "a1", "time"),
      link("a1", "a2", "data", "slow"),
    ],
    revealOrder: ["b-t", "test", "no", "yes", "b-a", "a1", "a2", "why"],
  },
  {
    id: "addressing",
    eyebrow: "17 · SEGMENT STORE",
    title: "Identity folds in the parent, and the domain",
    lede:
      "Three id functions over blake2b. parent_id is hashed first in all three, so an id encodes its whole ancestor chain. Domain tags keep the three from ever aliasing.",
    narration:
      "Segment identity is content addressing with two twists. First, the parent id is the very first field hashed, in all three functions — so an id transitively encodes its entire ancestor chain. Identical role and tokens under a different conversation prefix hash differently. That is what makes segment identity the same thing as cache-prefix identity. Second, domain separation: the text and raw variants carry distinct tag bytes, so a text-derived id can never alias a token-derived one. That matters more than it looks, because insertion is insert-if-absent — a collision would silently discard the second segment and ship the wrong message.",
    caption:
      "segment_ir/pool.py:37 segment_id, :62 text_segment_id with tag text, :81 raw_segment_id with tag raw. Bulk token encoding rationale at :44-49 — a per-token encode cost roughly 44 million calls on a corpus-scale trace. Insert-if-absent at :116, :130, :147.",
    nodes: [
      band("b-c", "TWO CONVERSATIONS", { col: 0, row: 0.6 }),
      card("r1", "shared prefix", "same parent chain", "done", { col: 0, row: 0.6 }),
      card("a1", "conversation A tail", "diverges", "data", { col: 1.3, row: 0 }),
      card("b1", "conversation B tail", "diverges", "time", { col: 1.3, row: 1.2 }),
      chip("dedup", "one stored span", { col: 2.5, row: 0.6 }),

      band("b-d", "THREE DOMAINS, NEVER ALIASING", { col: 0, row: 2.2 }),
      card("d1", "tokens", "untagged domain", "control", { col: 0, row: 2.2 }),
      card("d2", "text", "tag: text", "data", { col: 1.15, row: 2.2 }),
      card("d3", "raw", "tag: raw - verbatim wire", "muted", { col: 2.3, row: 2.2 }),

      note(
        "collide",
        "A collision would be silent",
        "Insertion is insert-if-absent, so an aliased id discards the second segment and materializes the first in its place. Wrong bytes on the wire, no error anywhere.",
        { col: 3.5, row: 1.4 },
      ),
    ],
    edges: [
      link("r1", "a1", "data"),
      link("r1", "b1", "time"),
      link("r1", "dedup", "done", "slow"),
    ],
    revealOrder: ["b-c", "r1", "a1", "b1", "dedup", "b-d", "d1", "d2", "d3", "collide"],
  },
  {
    id: "layout",
    eyebrow: "18 · SEGMENT STORE",
    title: "Four files, two different index shapes",
    lede:
      "Content is dense and index-addressed: a packed array of offset and size pairs where the handle is the array index. Nodes are sparse and string-keyed, so their index is JSON.",
    narration:
      "The store is four files, and the asymmetry between them is the interesting part. Content is one long blob with a packed binary index — alternating offset and size words, where a handle is literally the array index into it. Segment ids never appear on disk at all. Node envelopes are the opposite: sparse and string-keyed, so their index is a two-level JSON map from trace, to ordinal and phase variant, to a span. Dense and integer-addressed on one side, sparse and string-keyed on the other, because that is what each actually is.",
    caption:
      "graph_segment_unified_store.py:139-142 the four paths; content.idx is a packed array of unsigned long long built at :345-349, handle h spanning words 2h and 2h+1. nodes.idx is orjson at :351. Inner key is _encode_inner_key:35, called on both the write and read sides so the format is never duplicated.",
    nodes: [
      band("b-c", "CONTENT · DENSE", { col: 0, row: 0.3 }),
      card("cb", "content.blob", "every unique segment", "data", { col: 0, row: 0.3 }),
      card("ci", "content.idx", "packed offset/size words", "data", { col: 1.3, row: 0.3 }),
      chip("h", "handle = array index", { col: 2.6, row: 0.3 }),

      band("b-n", "NODES · SPARSE", { col: 0, row: 1.9 }),
      card("nb", "nodes.blob", "per-node envelopes", "control", { col: 0, row: 1.9 }),
      card("ni", "nodes.idx", "JSON two-level map", "control", { col: 1.3, row: 1.9 }),
      chip("k", "trace -> ord:variant", { col: 2.6, row: 1.9 }),

      card("client", "one client", "resolves both", "done", { col: 3.7, row: 1.1 }),
      note(
        "spill",
        "Only content spills incrementally",
        "The blob streams through a single open writer as it is built. The manifest region stays in RAM — it fills below the parse peak, so a second write handle would buy no peak-memory win.",
        { col: 0, row: 2.9 },
      ),
    ],
    edges: [
      link("cb", "ci", "data"),
      link("ci", "h", "data"),
      link("nb", "ni", "control"),
      link("ni", "k", "control"),
      ...fanIn(["h", "k"], "client", "done", "slow"),
    ],
    revealOrder: ["b-c", "cb", "ci", "h", "b-n", "nb", "ni", "k", "client", "spill"],
  },
  {
    id: "reader",
    eyebrow: "19 · SEGMENT STORE",
    title: "Assemble the body straight from mapped pages",
    lede:
      "Each stored segment is already valid JSON, so a request body is built by concatenating memoryview slices — no per-segment parse, no re-encode.",
    narration:
      "The read side is where the layout pays off. Both blobs are memory-mapped read-only, and the file descriptor is closed immediately — the mapping keeps the pages alive without holding it. Then the trick: every stored segment is already a valid JSON object, so building a request body is byte concatenation. Open bracket, a mapped span, a comma, another mapped span, close. No per-segment parse and no re-encode of the array, just one join at the end. Spans are validated at open rather than at read, because Python slice clamping would otherwise return quietly truncated bytes from a stale store.",
    caption:
      "graph_segment_unified_store.py:433-441 _map_blob closes the fd after mmap; :510-526 build_request_body_handles interleaves raw memoryview slices with the literal head at :26. _validate_content_spans:443-453 fails loud at open — otherwise slice clamping ships short prompts silently.",
    nodes: [
      band("b-m", "MAPPED ONCE", { col: 0, row: 1 }),
      card("mmap", "mmap ACCESS_READ", "fd closed straight after", "muted", { col: 0, row: 1 }),

      band("b-b", "BODY ASSEMBLY", { col: 1.3, row: -0.2 }),
      card("head", "head bytes", "messages open bracket", "muted", { col: 1.3, row: -0.2 }),
      card("s1", "span", "mapped page slice", "data", { col: 1.3, row: 0.7 }),
      card("s2", "span", "mapped page slice", "data", { col: 1.3, row: 1.6 }),
      card("tail", "tail bytes", "close bracket", "muted", { col: 1.3, row: 2.5 }),
      chip("join", "one join", { col: 2.6, row: 1.15 }),
      card("body", "request body", "one copy, total", "done", { col: 3.2, row: 1.15 }),

      note(
        "guard",
        "Spans validated at open",
        "Slice clamping would return silently truncated bytes from a stale or partial store, so the client fails loud at open instead of shipping a short prompt.",
        { col: 0, row: 2.6 },
      ),
    ],
    edges: [
      link("mmap", "s1", "muted", "slow"),
      link("mmap", "s2", "muted", "slow"),
      ...fanIn(["head", "s1", "s2", "tail"], "join", "data"),
      link("join", "body", "done"),
    ],
    revealOrder: ["b-m", "mmap", "b-b", "head", "s1", "s2", "tail", "join", "body", "guard"],
  },
  {
    id: "builder",
    eyebrow: "20 · SEGMENT STORE",
    title: "Two drains, one store shape, one abort",
    lede:
      "A dynamo trace without max_isl streams through a worker pool; everything else parses eagerly. Both converge on the same store, and both are wrapped in the same cleanup.",
    narration:
      "The builder has two routes. A Dynamo trace with no max-isl override streams through a worker pool, building the store incrementally. Everything else — native formats, and Dynamo when max-isl is set, because that option changes lowering semantics — parses once off the loop and interns the whole pool at the end. Both routes converge on an identical store shape. Both are wrapped in the same failure handler: abort the store, unlink its files, remove the directory, re-raise. And abort is guarded, so a store that already finalized is never deleted by a later failure.",
    caption:
      "store_build.py:185-301 the two drains; the max_isl carve-out is stated at :266-267. Abort at graph_segment_unified_store.py:298, guarded by the finalized flag at :316 so a finalized store survives. Ordinals come from flat_trie_ordinals, the same function the dispatch-time catalog calls — store_builder.py:32-39.",
    nodes: [
      band("b-in", "ONE INPUT", { col: 0, row: 1.1 }),
      card("src", "workload source", "format + options", "muted", { col: 0, row: 1.1 }),
      chip("fork", "dynamo && no max_isl?", { col: 1.2, row: 1.1 }),

      card("stream", "streaming drain", "worker pool, incremental", "control", { col: 2, row: 0.2 }),
      card("eager", "eager drain", "parse once, intern pool", "data", { col: 2, row: 2 }),
      card("store", "one store shape", "identical either way", "done", { col: 3.3, row: 1.1 }),

      band("b-x", "FAILURE", { col: 0, row: 2.7 }),
      card("abort", "abort + rmtree", "then re-raise", "failure", { col: 0, row: 2.7 }),
      chip("fin", "unless finalized", { col: 1.3, row: 2.7 }),
      note(
        "ord",
        "One ordinal function, both planes",
        "Build-time manifest ordinals and dispatch-time catalog ordinals come from the same call, which is what makes the worker read the right envelope.",
        { col: 2.3, row: 2.9 },
      ),
    ],
    edges: [
      link("src", "fork", "muted"),
      link("fork", "stream", "control"),
      link("fork", "eager", "data"),
      ...fanIn(["stream", "eager"], "store", "done"),
      link("abort", "fin", "failure"),
    ],
    revealOrder: ["b-in", "src", "fork", "stream", "eager", "store", "b-x", "abort", "fin", "ord"],
  },
  {
    id: "envelope",
    eyebrow: "21 · SEGMENT STORE",
    title: "What the worker actually reads",
    lede:
      "An envelope is handles plus dispatch overrides plus a stream flag. Optional keys are omitted entirely when unset, so envelopes stay byte-identical across corpora.",
    narration:
      "Finally, the envelope — the per-node record the worker reads at dispatch. It carries segment handles, dispatch overrides, and a stream flag, plus a few optional keys. Those optional keys are omitted rather than written as null, deliberately, so that a corpus with no headers produces byte-identical envelopes to one that never had the field. The worker looks up by trace, ordinal and phase variant, and rebuilds the request. Note the split: extra headers go to the request headers, never the body, and a hand-authored extra-body entry always wins over a folded-in default.",
    caption:
      "store_builder.py:152 _trie_envelope; the fold-only-if-absent rule at :174-180 and the byte-identity invariant stated twice, at :170-172 and store :244-246. Worker rebuild is worker_materialize.py:62 read_node_envelope, then the bytes fast path at :251.",
    nodes: [
      band("b-e", "ENVELOPE", { col: 0, row: 0.6 }),
      card("h", "handles", "into content.blob", "data", { col: 0, row: 0.2 }),
      card("ov", "dispatch_overrides", "model, max tokens, tools", "control", { col: 0, row: 1.2 }),
      card("st", "stream", "bool", "muted", { col: 0, row: 2.2 }),

      chip("key", "trace / ordinal / variant", { col: 1.3, row: 1.2 }),
      card("wk", "worker", "rebuilds the request", "done", { col: 2.1, row: 1.2 }),
      card("body", "body", "from handles", "data", { col: 3.3, row: 0.5 }),
      card("hdr", "headers", "never the body", "time", { col: 3.3, row: 1.9 }),

      note(
        "omit",
        "Omitted, not null",
        "Optional keys are left out entirely when unset, so a header-less corpus produces envelopes byte-identical to one that never carried the field.",
        { col: 0, row: 3.1 },
      ),
    ],
    edges: [
      ...fanIn(["h", "ov", "st"], "key", "data"),
      link("key", "wk", "control"),
      link("wk", "body", "data"),
      link("wk", "hdr", "time"),
    ],
    revealOrder: ["b-e", "h", "ov", "st", "key", "wk", "body", "hdr", "omit"],
  },
  {
    id: "firing",
    eyebrow: "22 · THE EXECUTOR",
    title: "One task per node, four ways out",
    lede:
      "Every node is an asyncio Task in a TaskGroup. Awaiting inputs, then the firing gate, then dispatch — and four distinct exits, only one of them normal.",
    narration:
      "Now the runtime. Each node becomes one asyncio task inside a task group, so the group exit is the join — there is no queue and no gather anywhere in this engine. A node awaits its inputs, sleeps out its firing gate, dispatches, and executes. What is worth studying is the exit. A clean overflow stop ends the trajectory without scheduling successors. A dispatch failure is contained: it writes a type-correct sentinel so downstream fan-in still unblocks. A refusal is re-raised. And the finally block always runs.",
    caption:
      "graph/executor.py:219 _drive_frontier TaskGroup; :278 _fire; :345-424 the containment branch, which re-raises refusal and stickiness before the generic case; sentinel is [] for MESSAGES channels because add_messages rejects non-lists.",
    nodes: [
      band("b-l", "ONE TASK PER NODE", { col: 0, row: 0.9 }),
      card("await", "await inputs", "channel arrivals", "control", { col: 0, row: 0.9 }),
      card("gate", "firing gate", "clock sleep", "time", { col: 1.1, row: 0.9 }),
      card("exec", "execute", "dispatch to worker", "data", { col: 2.2, row: 0.9 }),

      band("b-x", "FOUR EXITS", { col: 3.4, row: -0.8 }),
      card("x1", "normal", "publish + successors", "done", { col: 3.4, row: -0.8 }),
      card("x2", "overflow terminate", "clean stop, no successors", "muted", { col: 3.4, row: 0.3 }),
      card("x3", "dispatch failure", "sentinel write, successors still fire", "time", { col: 3.4, row: 1.4 }),
      card("x4", "refusal / stickiness", "re-raised, trace fatal", "failure", { col: 3.4, row: 2.5 }),

      chip("fin", "finally: finalize", { col: 2.3, row: 2.5 }),
    ],
    edges: [
      link("await", "gate", "control"),
      link("gate", "exec", "time"),
      ...fanOut("exec", ["x1", "x2", "x3", "x4"], "control"),
      link("exec", "fin", "muted", "slow"),
    ],
    revealOrder: ["b-l", "await", "gate", "exec", "b-x", "x1", "x2", "x3", "x4", "fin"],
  },
  {
    id: "dualfan",
    eyebrow: "23 · DUAL FAN-OUT",
    title: "A node fans out twice, at different moments",
    lede:
      "Start-anchored children launch at the parent's dispatch and run alongside it. Completion children launch after. Two separate adjacency maps keep them apart.",
    narration:
      "This is the shape that makes the engine unusual. A node fans out twice. Start-anchored successors are scheduled the instant the parent dispatches, so they run concurrently with a parent that is still streaming. Completion successors are scheduled when it finishes. The scheduler keeps these in two separate maps on purpose — if a start-anchored child were also in the completion map it would be re-scheduled and trip the cycle guard.",
    caption:
      "graph/executor.py:271 schedules start_anchored_successors inside _prepare_node_inputs, at dispatch; :449 _schedule_successors at completion. scheduler.py:93 keeps _start_anchored_succ separate from _static_succ; :104 rejects mixed-anchor fan-in outright.",
    nodes: [
      card("parent", "parent node", "still streaming", "control", { col: 1.2, row: 1 }),
      band("b-d", "AT DISPATCH", { col: 2.5, row: -0.2 }),
      card("sa1", "start-anchored child", "runs alongside", "time", { col: 2.5, row: -0.2 }),
      card("sa2", "start-anchored child", "runs alongside", "time", { col: 2.5, row: 0.8 }),

      band("b-c", "AT COMPLETION", { col: 2.5, row: 2 }),
      card("cc1", "completion child", "after the reply", "done", { col: 2.5, row: 2 }),
      card("cc2", "completion child", "after the reply", "done", { col: 2.5, row: 3 }),

      note(
        "sep",
        "Why two maps",
        "A start-anchored child in the completion map would be scheduled twice and trip the cycle guard. Mixed-anchor fan-in is rejected at construction.",
        { col: 0, row: 2.4 },
      ),
    ],
    edges: [
      ...fanOut("parent", ["sa1", "sa2"], "time"),
      ...fanOut("parent", ["cc1", "cc2"], "done", "slow"),
    ],
    revealOrder: ["parent", "b-d", "sa1", "sa2", "b-c", "cc1", "cc2", "sep"],
  },
  {
    id: "lanes",
    eyebrow: "24 · CONCURRENCY",
    title: "Lanes with a feedback loop, not a queue",
    lede:
      "A fixed pool of lanes each loop: draw a template, run it, release, draw again. Two terminal conditions, and a bare run must not recycle forever.",
    narration:
      "Cross-trace concurrency is a fixed pool of lanes. Each lane loops — draw a template, run it to completion, release the slot, draw the next. The loop is the point: this is genuinely cyclic, not a pipeline. There are two ways it ends. With an explicit stop condition it recycles until that condition trips. Without one, each lane draws until every corpus position has been claimed exactly once, then stops, because a bare graph run must not recycle forever.",
    caption:
      "timing/strategies/graph_ir_replay.py:1259 _run_lanes; :1532 _recycle_has_stop_condition selects bounded vs single-pass; :857 _draw_index — random is coerced to shuffle because each recycle is one corpus pass.",
    nodes: [
      band("b-l", "LANE POOL", { col: 1.3, row: 0.9 }),
      card("corpus", "trace corpus", "N templates", "data", { col: 0, row: 0.9 }),
      card("l0", "lane 0", "draw - run - release", "control", { col: 1.3, row: 0 }),
      card("l1", "lane 1", "draw - run - release", "control", { col: 1.3, row: 0.9 }),
      card("l2", "lane N", "draw - run - release", "control", { col: 1.3, row: 1.8 }),
      chip("draw", "draw_index", { col: 2.6, row: 0.9 }),

      band("b-e", "TWO TERMINAL CONDITIONS", { col: 0, row: 2.8 }),
      card("t1", "stop condition set", "recycle until it trips", "done", { col: 0, row: 2.8 }),
      card("t2", "no stop condition", "single pass, then stop", "muted", { col: 1.4, row: 2.8 }),
      note(
        "loop",
        "The feedback arrow is the design",
        "A released lane re-draws onto its own freed slot, so concurrency stays pinned at the lane count rather than draining.",
        { col: 3, row: 2.4 },
      ),
    ],
    edges: [
      ...fanOut("corpus", ["l0", "l1", "l2"], "data"),
      ...fanIn(["l0", "l1", "l2"], "draw", "control"),
      link("draw", "l1", "time", "slow"),
    ],
    revealOrder: ["b-l", "corpus", "l0", "l1", "l2", "draw", "b-e", "t1", "t2", "loop"],
  },
  {
    id: "demux",
    eyebrow: "25 · RETURNS",
    title: "One observer, de-multiplexed to a parked Future",
    lede:
      "No queue. A return routes observer to adapter to the Future parked for one correlation id and turn, while first tokens travel a parallel track to an Event.",
    narration:
      "Finally, how a reply gets back to the coroutine that is awaiting it. There is no queue. A single observer receives every return, looks up the adapter for that trace, and resolves the future parked under a correlation id and turn index. That wakes exactly the one node coroutine that dispatched it. First tokens travel a completely parallel track — a stamp closure that sets an event, which is what a first-token-anchored successor is waiting on. Two tracks, one fabric.",
    caption:
      "graph/credit_dispatch_adapter.py:265 dispatch parks loop.create_future(); :355 resolve is called synchronously from the credit callback on the same loop, which is why the waiter dict needs no lock. First-token track: dispatch/llm.py:45 _make_first_token_stamp.",
    nodes: [
      card("obs", "graph return observer", "one entry point", "control", { col: 0, row: 1 }),
      chip("lookup", "adapters[trace_id]", { col: 1.15, row: 1 }),
      band("b-w", "PARKED FUTURES", { col: 2.1, row: 0 }),
      card("f1", "future (corr, turn 0)", "awaiting _fire", "data", { col: 2.1, row: 0 }),
      card("f2", "future (corr, turn 1)", "awaiting _fire", "data", { col: 2.1, row: 1 }),
      card("f3", "future (corr, turn 2)", "awaiting _fire", "data", { col: 2.1, row: 2 }),

      band("b-ft", "PARALLEL TRACK", { col: 0, row: 2.7 }),
      card("ft", "first token", "stamp closure", "time", { col: 0, row: 2.7 }),
      card("ev", "Event.set()", "wakes gated successor", "done", { col: 1.3, row: 2.7 }),

      note(
        "lock",
        "No lock on the waiter map",
        "resolve runs synchronously from the credit callback on the same loop, so the dict is thread-confined by construction.",
        { col: 3.4, row: 1.2 },
      ),
    ],
    edges: [
      link("obs", "lookup", "control"),
      ...fanOut("lookup", ["f1", "f2", "f3"], "data"),
      link("ft", "ev", "time"),
    ],
    revealOrder: ["obs", "lookup", "b-w", "f1", "f2", "f3", "b-ft", "ft", "ev", "lock"],
  },
];

/** Registered in `App.tsx`; served by the generic `DeckRoute` at `/python-graph-workload`. */
export const PYTHON_GRAPH_WORKLOAD_DECK: DeckDefinition = {
  id: "python-graph-workload",
  title: "Python Graph Workload: trace, build, dataflow",
  slides: SLIDES,
};
