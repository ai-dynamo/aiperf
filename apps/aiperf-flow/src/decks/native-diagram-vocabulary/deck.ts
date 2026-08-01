/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the native diagram vocabulary added to this app: six node types that
//! let a narrated slide state things box-and-arrow diagrams cannot.
//!
//! The deck is its own demo. Every chart below is the real node type rendering real data — the
//! same six requests and the same recorded weka trace carried from slide to slide, so the viewer
//! watches one dataset become progressively more expressible as each node type lands.

import type { DeckDefinition, SlideDefinition } from "../../deck/types.js";
import {
  band,
  blocks,
  card,
  chip,
  fanOut,
  intervals,
  link,
  note,
  ragged,
  slices,
  sweep,
  timeline,
} from "../shared/diagram.js";
import type { SweepRequest } from "../../nodes/sweepMath.js";

/** The running example. Carried across every slide so each new node type re-reads one dataset. */
const REQUESTS: SweepRequest[] = [
  { id: "A", start: 0, gen: 6, end: 20, tokens: 120 },
  { id: "B", start: 3, gen: 10, end: 30, tokens: 200 },
  { id: "C", start: 8, gen: 12, end: 24, tokens: 90 },
  { id: "D", start: 14, gen: 22, end: 40, tokens: 260 },
  { id: "E", start: 18, gen: 25, end: 35, tokens: 150 },
  { id: "F", start: 28, gen: 34, end: 50, tokens: 180 },
];

/** A recorded weka trace, warped at cap 5s. Bars are the ActiveIdleWarp sweep's actual output. */
const WARP_BARS = [
  { id: "m0", lane: "main", rawStart: 0, rawEnd: 3, warpStart: 0, warpEnd: 3 },
  { id: "m1", lane: "main", rawStart: 90, rawEnd: 93, warpStart: 8, warpEnd: 11 },
  { id: "a0", lane: "sub-A", rawStart: 94, rawEnd: 102, warpStart: 12, warpEnd: 20 },
  { id: "b0", lane: "sub-B", rawStart: 96, rawEnd: 106, warpStart: 14, warpEnd: 24 },
  { id: "a1", lane: "sub-A", rawStart: 102, rawEnd: 107, warpStart: 20, warpEnd: 25 },
  { id: "b1", lane: "sub-B", rawStart: 106, rawEnd: 110, warpStart: 24, warpEnd: 28 },
  { id: "m2", lane: "main", rawStart: 120, rawEnd: 122, warpStart: 33, warpEnd: 35 },
];

const SLIDES: readonly SlideDefinition[] = [
  {
    id: "before",
    eyebrow: "01 · BEFORE",
    title: "Six requests, and nothing true to say about them",
    lede:
      "The old vocabulary was header, panel, chip, and card. Every one is a box. A box can name a request; it cannot say when the request ran.",
    narration:
      "Here is the starting point. The deck engine had four node types — header, panel, chip, and card — and all four are boxes joined by arrows. Watch what happens when six real requests arrive. Each becomes a card, and each card is honest as far as it goes: it carries an id, and a duration written out as text. But look at what is missing. Nothing on this slide says that A and B were running at the same time. Nothing says the gap between C ending and F starting was mostly idle. Those are claims about a shared axis, and a box has no axis. You can write the numbers into the card and hope the reader does the arithmetic, which is what every deck here was doing.",
    caption:
      "src/nodes/nodeTypes.ts registered four types before this change: header, panel, chip, card. All are DOM boxes positioned on a grid by src/decks/shared/diagram.ts.",
    nodes: [
      band("b-old", "THE OLD VOCABULARY", { col: 0, row: -0.5 }),
      ...REQUESTS.map((r, i) =>
        card(
          `c-${r.id}`,
          `request ${r.id}`,
          `${r.start} → ${r.end}`,
          i % 2 === 0 ? "data" : "control",
          { col: (i % 3) * 1.05, row: 0.4 + Math.floor(i / 3) * 1.1 },
        ),
      ),
      note(
        "gap",
        "What a box cannot say",
        "A and B overlap. C and F do not. The cards above are identical in shape either way — the reader is being asked to sort six intervals in their head.",
        { col: 3.4, row: 0.9 },
      ),
    ],
    edges: [],
    revealOrder: [
      "b-old", "c-A", "c-B", "c-C", "c-D", "c-E", "c-F", "gap",
    ],
  },
  {
    id: "timeline",
    eyebrow: "02 · THE FIRST NODE TYPE",
    title: "timeline — put them on an axis",
    lede:
      "The same kind of data, now time-scaled across swimlanes on two clocks. Overlap is visible instead of inferred.",
    narration:
      "The first node type is timeline. Give it lanes and bars and it draws a time-scaled swimlane chart — and now the claims that were unsayable are simply visible. This is a recorded agent trace rather than the six requests, because it shows the second thing a shared axis buys you: two clocks at once. The top block is what was recorded. Eighty-seven seconds of dead air sit between the first request and the burst that follows. The bottom block is what the runtime actually replays, with that dead air collapsed to a five second cap. Every bar keeps its exact width, because service time is never compressed, and sub-A and sub-B still overlap by exactly as much as they did. You are not being told those two things are true. You can see them.",
    caption:
      "src/nodes/Timeline.tsx with geometry in timelineLayout.ts. Bars are the ActiveIdleWarp sweep's real output at cap=5s: m1 90→8, a0 94→12, m2 120→33.",
    nodes: [
      timeline(
        "tl",
        {
          title: "Recorded agent trace, warped at cap 5s",
          lanes: ["main", "sub-A", "sub-B"],
          bars: WARP_BARS,
          gaps: [
            { start: 3, end: 90, idle: 87, capped: true },
            { start: 93, end: 94, idle: 1, capped: false },
            { start: 110, end: 120, idle: 10, capped: true },
          ],
        },
        { col: 0, row: 0 },
      ),
      chip("k1", "overlap: visible", { col: 0, row: 2.9 }),
      chip("k2", "idle: measured", { col: 0.8, row: 2.9 }),
      chip("k3", "warp: comparable", { col: 1.6, row: 2.9 }),
      note(
        "how",
        "122 seconds replay in 35",
        "Only true dead air is cut — running-max end to next start. The 1-second gap is under the cap and survives untouched; the 87 and 10 second gaps are cut to 5.",
        { col: 2.5, row: 2.6 },
      ),
    ],
    edges: [],
    revealOrder: ["tl", "k1", "k2", "k3", "how"],
  },
  {
    id: "intervals",
    eyebrow: "03 · SECOND NODE TYPE",
    title: "intervals — badge the quantity the rule reads",
    lede:
      "Finished-before is a partial order. Rank, the position in sort(start, end, id), is the total order that breaks the ties.",
    narration:
      "The second node type is intervals. It looks like a Gantt, but the badge on each bar's end is the point. Edge derivation asks two questions about any pair: did A end before B started, and does A outrank B. Both quantities are now on screen — the bar's right edge answers the first, the badge answers the second — so a reader can evaluate the rule themselves instead of trusting a sentence about it. Rank is derived here from the same sort the real code uses, start then end then id, which is why Explore one carries badge one even though it is drawn second. And the dashed bar is the exclusion: an async-launched interval can finish first and still never become anyone's predecessor.",
    caption:
      "src/nodes/Intervals.tsx; intervalRanks() in intervalsLayout.ts derives the badge from sort(start, end, id) unless a row overrides it.",
    nodes: [
      intervals(
        "iv",
        {
          title: "Intervals on the warped clock — rank badge on each end",
          rows: [
            { id: "P0", label: "parent", start: 0, end: 1.0, role: "blue" },
            { id: "A0", label: "Explore #1", start: 1.2, end: 4.0, role: "green" },
            { id: "B0", label: "Explore #2", start: 1.3, end: 5.0, role: "green", dashed: true },
            { id: "C0", label: "spawned", start: 5.2, end: 7.0, role: "purple" },
            { id: "P1", label: "parent resume", start: 7.5, end: 8.0, role: "blue" },
          ],
        },
        { col: 0, row: 0 },
      ),
      chip("q1", "ended before?", { col: 0, row: 2.2 }),
      chip("q2", "outranks?", { col: 0.85, row: 2.2 }),
      note(
        "dash",
        "The dashed exclusion",
        "B0 ends at 5.0 before C0 starts at 5.2, and outranks it — and is still not a predecessor, because an async leaf is dropped for any target outside its subtree.",
        { col: 1.8, row: 2.0 },
      ),
    ],
    edges: [],
    revealOrder: ["iv", "q1", "q2", "dash"],
  },
  {
    id: "blocks",
    eyebrow: "04 · THIRD NODE TYPE",
    title: "blocks — compare two paths cell by cell",
    lede:
      "Twenty-three blocks of shared prefix. One relabelled block, and the segment-id chain diverges from there on.",
    narration:
      "The third node type is blocks, and it exists for arguments about prefixes. A prefix is only reusable if both paths agree on every block, so the claim is a cell-by-cell comparison — which means the honest way to make it is to draw every cell. On the left is the old per-turn scheme: advancing a turn relabelled block twenty on the parent chain but not on the forking subagent. One block out of twenty-three. On the right, with the tag frozen at creation, the two strips are identical. The reason one block matters is that segment ids are prefix-dependent: change block twenty's tag and every id after it changes too, so a twenty-three block prefix quietly degrades to a twenty block one.",
    caption:
      "src/nodes/Blocks.tsx. The divergence is the receipt block 57f2a77e; freezing rationale at trie_content.py:829-836.",
    nodes: [
      blocks(
        "old",
        {
          title: "per-turn tagging (old)",
          strips: [
            {
              label: "parent chain",
              cells: Array.from({ length: 23 }, (_, i) => (i < 20 ? "blue" : "purple")),
            },
            { label: "forking subagent", cells: Array.from({ length: 23 }, () => "blue") },
          ],
          highlight: 20,
          detail: "Block 20 disagrees — different tokenization, cache miss from there on.",
        },
        { col: 0, row: 0 },
      ),
      blocks(
        "new",
        {
          title: "frozen per-block (new)",
          strips: [
            { label: "parent chain", cells: Array.from({ length: 23 }, () => "blue") },
            { label: "forking subagent", cells: Array.from({ length: 23 }, () => "blue") },
          ],
          highlight: 20,
          detail: "Identical strips — both paths emit the same segment-id chain, prefix hits.",
        },
        { col: 1.7, row: 0 },
      ),
      note(
        "cost",
        "23 blocks becomes 20",
        "Segment ids are prefix-dependent, so a single differing tag invalidates itself and everything after it.",
        { col: 0.6, row: 1.9 },
      ),
    ],
    edges: [],
    revealOrder: ["old", "new", "cost"],
  },
  {
    id: "sweep",
    eyebrow: "05 · FOURTH NODE TYPE",
    title: "sweep — the six requests, and the curve they generate",
    lede:
      "Back to the opening dataset. Each interval contributes +weight at its start and −weight at its end; a running cumsum over those events is exactly the curve.",
    narration:
      "The fourth node type brings back the six requests from the first slide — the ones that were six identical boxes. Here they are on an axis, and beneath them the concurrency curve they produce. The relationship between the two halves is the whole point: every bar's left edge lines up with a green tick and a step up, every right edge with an orange tick and a step down. That is the sweep-line identity made checkable by eye rather than asserted. Sorting those events costs E log E, and no part of the timeline is ever scanned. Changing which curve you want changes only the weight — one for concurrency, output tokens for tokens in flight, tokens per decode second for throughput.",
    caption:
      "src/nodes/Sweep.tsx over sweepMath.ts, moved out of the metrics-accumulator deck so the node and that deck's interactive view draw the same math.",
    nodes: [
      sweep(
        "sw",
        {
          title: "Concurrency — the same six requests from slide 1",
          requests: REQUESTS,
          curve: "concurrency",
          tMax: 52,
          axisLabel: "green = +delta at a start · orange = −delta at an end",
          valueLabel: "requests",
        },
        { col: 0, row: 0 },
      ),
      chip("w1", "weight 1 → concurrency", { col: 0, row: 2.6 }),
      chip("w2", "weight tokens → in-flight", { col: 1.0, row: 2.6 }),
      chip("w3", "weight tok/s → throughput", { col: 2.0, row: 2.6 }),
    ],
    edges: [],
    revealOrder: ["sw", "w1", "w2", "w3"],
  },
  {
    id: "slices",
    eyebrow: "06 · FIFTH NODE TYPE",
    title: "slices — same six requests, now bucketed",
    lede:
      "A uniform grid bins each record by its start. The grid runs to 60 while activity ends at 50, so the trailing bucket is clipped and flagged.",
    narration:
      "The fifth node type takes those same six requests one more time and lays a uniform grid over them. Two things this makes visible. First, binning is by start — the dot on each bar — not by overlap, so B spans three slices and still belongs to slice zero alone. Second, look at the right edge. Activity ends at fifty, but a fifteen-unit grid runs to sixty, and that orange band is ten units of nothing. A rate computed over slice three's full width would be diluted by that padding, so the slice is clipped to real activity and starred. Same dataset, third view of it, and each view answers a question the others cannot.",
    caption:
      "src/nodes/Slices.tsx; buildSlices() in slicesLayout.ts flags is_complete=false rather than dropping the partial bucket, so its records survive and its width is honest.",
    nodes: [
      slices(
        "sl",
        {
          title: "slice_duration = 15 — grid to 60, activity to 50",
          requests: REQUESTS.map((r) => ({ id: r.id, start: r.start, end: r.end })),
          duration: 15,
          axisLabel: "dot = record start (the binning key) · orange = grid overrun",
        },
        { col: 0, row: 0 },
      ),
      chip("s1", "bin by start", { col: 0, row: 2.3 }),
      chip("s2", "count once", { col: 0.75, row: 2.3 }),
      note(
        "trail",
        "Flag it, do not drop it",
        "Dropping the partial bucket loses real records; keeping it at full width dilutes the rate. Clipping plus is_complete=false does neither.",
        { col: 1.6, row: 2.1 },
      ),
    ],
    edges: [],
    revealOrder: ["sl", "s1", "s2", "trail"],
  },
  {
    id: "ragged",
    eyebrow: "07 · SIXTH NODE TYPE",
    title: "ragged — the shape underneath",
    lede:
      "Variable-length per-record lists packed into values, record_indices, and offsets — with −1 for a record that contributed nothing.",
    narration:
      "The last node type draws a memory layout rather than a timeline. Inter-chunk latency is a list per record, and each list is a different length. Packed flat, one values array holds every element, record-indices names each element's owner, and offsets says where each record's run begins. Watch record one: it streamed as a single chunk, so it has no gaps at all, and its offset is minus one rather than a real index — because a real index there would be indistinguishable from record two's start. Absent and empty are different states and the layout keeps them apart. Once the data is shaped this way, a per-record question is a mask over one column instead of a loop over records.",
    caption:
      "src/nodes/Ragged.tsx; flattenRagged() in raggedLayout.ts. Same inter_chunk_latency fixture as the interactive metrics workbook.",
    nodes: [
      ragged(
        "rg",
        {
          title: "inter_chunk_latency — five records",
          lists: [[12, 9, 11], [], [8, 10, 9, 13, 7], [10, 12], [9]],
          highlight: 2,
          raggedLabel: "per-record lists (ragged)",
          flatLabel: "flat arrays (one allocation, masked in bulk)",
        },
        { col: 0, row: 0 },
      ),
      chip("r1", "−1 ≠ 0", { col: 0, row: 2.4 }),
      note(
        "mask",
        "Why flatten",
        "One allocation instead of N, and a dataset-wide per-record query becomes a comparison over record_indices — no Python-level loop.",
        { col: 0.9, row: 2.2 },
      ),
    ],
    edges: [],
    revealOrder: ["rg", "r1", "mask"],
  },
  {
    id: "seam",
    eyebrow: "08 · THE SEAM",
    title: "What adding one actually takes",
    lede:
      "A pure layout module, a component, one registry line, one authoring helper. The layout module exists so the node can declare its box before React Flow measures it.",
    narration:
      "Finally, the shape of the change itself, because the next one should be cheap. Each node type is four pieces. A pure layout module holds the geometry and any derivation, which makes it unit-testable without rendering. A component reads that layout and emits SVG, ending with the anchor handles so edges can still attach. One line registers it in the node-types map. And one authoring helper lets a deck place it on the grid. The reason geometry is split out rather than living in the component is specific: the slide re-fits the view every time another node is revealed, so a node that measures itself late makes the whole diagram reframe mid-cascade. The helper sets width and height from the same function the component draws with, and the two cannot disagree.",
    caption:
      "src/nodes/{Timeline,Intervals,Blocks,Sweep,Slices,Ragged}.tsx with matching *Layout.ts; registered in nodeTypes.ts; authored via src/decks/shared/diagram.ts.",
    nodes: [
      band("b-4", "FOUR PIECES PER NODE TYPE", { col: 0, row: 0.5 }),
      card("lay", "xLayout.ts", "pure geometry + derivation", "data", { col: 0, row: 0.5 }),
      card("cmp", "X.tsx", "SVG + NodeAnchorHandles", "control", { col: 1.05, row: 0.5 }),
      card("reg", "nodeTypes.ts", "one line", "muted", { col: 2.1, row: 0.5 }),
      card("hlp", "diagram.ts", "authoring helper", "done", { col: 3.15, row: 0.5 }),

      chip("size", "style: {width, height}", { col: 1.6, row: 1.7 }),
      note(
        "why",
        "Why geometry is split out",
        "Slide re-fits the view on every reveal tick. A node measured late reframes the diagram mid-cascade, so the helper sizes it up front from the same function the component draws with.",
        { col: 0, row: 2.5 },
      ),
      note(
        "plain",
        "Plain data, so .ts decks work",
        "Every one of these takes arrays and numbers, not JSX. A declarative deck authors a chart without becoming a .tsx file.",
        { col: 1.6, row: 2.5 },
      ),
    ],
    edges: [
      link("lay", "cmp", "data"),
      link("cmp", "reg", "control"),
      link("reg", "hlp", "muted"),
      ...fanOut("lay", ["size"], "time", "slow"),
      link("hlp", "size", "done", "slow"),
    ],
    revealOrder: [
      "b-4", "lay", "cmp", "reg", "hlp", "size", "why", "plain",
    ],
  },
];

/** Registered in `App.tsx`; served by the generic `DeckRoute` at `/native-diagram-vocabulary`. */
export const NATIVE_DIAGRAM_VOCABULARY_DECK: DeckDefinition = {
  id: "native-diagram-vocabulary",
  title: "What Changed: the native diagram vocabulary",
  slides: SLIDES,
};
