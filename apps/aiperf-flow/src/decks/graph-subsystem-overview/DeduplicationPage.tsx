/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deduplication page: how prompt dedup and the content-parent trie make prefix-cache reuse
//! provable offline. Ports `DedupVisual`, `ContentTrieVisual` (as React Flow), `PrefixCacheChart`
//! (as a composition table), `BlockGeometryVisual`, `BlockRoleChart`, `UnifiedStoreLayout`, and
//! `WarmupVariantVisual` from `graph-subsystem-overview.canvas.tsx`.

import { useState } from "react";
import clsx from "clsx";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Table } from "../../prose/Table.js";
import { Code } from "../../prose/Code.js";
import { Toggle } from "../../prose/Toggle.js";
import { Divider } from "../../layout/Divider.js";
import {
  inkClassName,
  strokeClassName,
  surfaceClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../../theme/tokens.js";
import type { Audience } from "./audience.js";

interface Block {
  key: string;
  label: string;
}
const REQ_A: Block[] = [
  { key: "sys", label: "system" },
  { key: "u1", label: "user 1" },
  { key: "a1", label: "asst 1" },
  { key: "u2", label: "user 2" },
];
const REQ_B: Block[] = [
  { key: "sys", label: "system" },
  { key: "u1", label: "user 1" },
  { key: "a1", label: "asst 1" },
  { key: "u3", label: "user 3" },
];

const SHARED = new Set(["sys", "u1", "a1"]);
function segId(b: Block): string {
  if (SHARED.has(b.key)) return `S${["sys", "u1", "a1"].indexOf(b.key)}`;
  return b.key === "u2" ? "S3a" : "S3b";
}

function DedupVisual(): React.JSX.Element {
  const [dedup, setDedup] = useState(true);
  const renderRow = (blocks: Block[], rowLabel: string) => (
    <Row gap={8} align="center">
      <div className={clsx("w-20 shrink-0 text-xs", inkClassName("tertiary"))}>{rowLabel}</div>
      <Row gap={6} wrap>
        {blocks.map((b) => {
          const shared = dedup && SHARED.has(b.key);
          const fresh = dedup && !SHARED.has(b.key);
          return (
            <div key={b.key} className="flex flex-col items-center gap-1">
              <div
                className={clsx(
                  "min-w-[62px] rounded-none border px-3 py-2 text-center text-xs font-semibold",
                  strokeClassName("secondary"),
                  shared ? clsx(categoryBgTintClassName("green"), inkClassName("primary")) : fresh ? clsx(categoryBgTintClassName("orange"), inkClassName("primary")) : clsx(surfaceClassName("panel"), inkClassName("primary")),
                )}
              >
                {b.label}
              </div>
              {dedup && <span className={clsx("text-xs", inkClassName("tertiary"))}>{segId(b)}</span>}
            </div>
          );
        })}
      </Row>
    </Row>
  );

  return (
    <div className={clsx("rounded-none border px-4 py-4", strokeClassName("primary"), surfaceClassName("elevated"))}>
      <Stack gap={14}>
        <Row align="center" gap={10}>
          <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{dedup ? "Segment-trie dedup" : "Naive replay"}</span>
          <div className="flex-1" />
          <span className={clsx("text-xs", inkClassName("tertiary"))}>Dedup</span>
          <Toggle checked={dedup} onChange={setDedup} />
        </Row>
        {renderRow(REQ_A, "request A")}
        {renderRow(REQ_B, "request B")}
        <Divider />
        {dedup ? (
          <p className={clsx("text-sm", inkClassName("secondary"))}>
            Both requests share the <strong>system → user 1 → asst 1</strong> prefix, so those blocks hash to the
            identical segment ids <Code inline>S0 · S1 · S2</Code>. Only the divergent tail is a fresh id. The server
            sees the same leading token sequence twice and serves it from its KV prefix cache.
          </p>
        ) : (
          <p className={clsx("text-sm", inkClassName("secondary"))}>
            Without content-addressed segments, each request is an independent blob. The shared prefix is re-sent
            verbatim and the server cannot tell that it repeats — no reuse is provable.
          </p>
        )}
      </Stack>
    </div>
  );
}

const TRIE_NODES: Node[] = [
  { id: "root", type: "card", position: { x: 200, y: 0 }, data: { title: "system", subtitle: "shared root", className: categoryBgTintClassName("purple") } },
  { id: "r1", type: "card", position: { x: 200, y: 120 }, data: { title: "req 1", subtitle: "prefix: S" } },
  { id: "r2", type: "card", position: { x: 40, y: 240 }, data: { title: "req 2", subtitle: "extends req 1" } },
  { id: "r3", type: "card", position: { x: 360, y: 240 }, data: { title: "req 3", subtitle: "branches at req 1" } },
  { id: "r4", type: "card", position: { x: 40, y: 360 }, data: { title: "req 4", subtitle: "extends req 2" } },
];
const TRIE_EDGES: Edge[] = [
  { id: "e-root-r1", source: "root", target: "r1", type: "flow" },
  { id: "e-r1-r2", source: "r1", target: "r2", type: "flow" },
  { id: "e-r1-r3", source: "r1", target: "r3", type: "flow" },
  { id: "e-r2-r4", source: "r2", target: "r4", type: "flow" },
];

function ContentTrie(): React.JSX.Element {
  return (
    <Stack gap={8}>
      <div style={{ height: 460 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={TRIE_NODES}
          edges={TRIE_EDGES}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        Each request&apos;s content parent is the earlier request whose block hashes are its longest full prefix
        (tie-break to the most recent). Shared prefixes merge into one trie path, so identical leading blocks produce
        identical segment ids.
      </p>
    </Stack>
  );
}

function BlockGeometry(): React.JSX.Element {
  const [prevOut, setPrevOut] = useState("2");
  const [parentHasUser, setParentHasUser] = useState(true);
  const prev = [10, 20, 30, 40];
  const curr = [10, 20, 30, 55, 66];
  const lcp = (() => {
    const n = Math.min(prev.length, curr.length);
    for (let i = 0; i < n; i++) if (prev[i] !== curr[i]) return i;
    return n;
  })();
  const mCovered = curr.length;
  const inherited = Math.min(lcp, prev.length, mCovered);
  const newN = Math.max(0, mCovered - inherited);
  let asst = parentHasUser ? Math.min(parseInt(prevOut, 10), newN) : 0;
  if (asst === newN && asst > 0) asst -= 1;
  const roles = [...Array(asst).fill("assistant"), ...Array(newN - asst).fill("user")];

  const chip = (label: string, tone: CategoryRole | "prev", tag?: string) => (
    <div className="flex flex-col items-center gap-1">
      <div
        className={clsx(
          "min-w-[40px] rounded-none border px-2.5 py-1.5 text-center text-[11px] font-semibold",
          strokeClassName("secondary"),
          tone === "prev" ? clsx(surfaceClassName("panel"), inkClassName("primary")) : clsx(categoryBgTintClassName(tone), inkClassName("primary")),
        )}
      >
        {label}
      </div>
      {tag && <span className={clsx("text-xs", inkClassName("tertiary"))}>{tag}</span>}
    </div>
  );

  return (
    <Stack gap={12}>
      <Row gap={16} wrap align="center">
        <Row gap={6} align="center">
          <span className={clsx("text-xs", inkClassName("tertiary"))}>prev_out blocks</span>
          {["0", "1", "2"].map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={prevOut === v}
              onClick={() => setPrevOut(v)}
              className={clsx("rounded-none border px-2.5 py-0.5 text-xs font-medium", strokeClassName(prevOut === v ? "primary" : "secondary"), prevOut === v ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}
            >
              {v}
            </button>
          ))}
        </Row>
        <Row gap={8} align="center">
          <Toggle checked={parentHasUser} onChange={setParentHasUser} />
          <span className={clsx("text-sm", inkClassName("secondary"))}>parent has user context</span>
        </Row>
      </Row>
      <Stack gap={8}>
        <Row gap={8} align="center">
          <div className={clsx("w-14 shrink-0 text-xs", inkClassName("tertiary"))}>parent</div>
          <Row gap={6}>{prev.map((h, i) => <div key={i}>{chip(String(h), "prev")}</div>)}</Row>
        </Row>
        <Row gap={8} align="center">
          <div className={clsx("w-14 shrink-0 text-xs", inkClassName("tertiary"))}>this turn</div>
          <Row gap={6}>
            {curr.map((h, i) => {
              const isInh = i < inherited;
              const role = roles[i - inherited];
              const tone: CategoryRole = isInh ? "cyan" : role === "assistant" ? "blue" : "green";
              return <div key={i}>{chip(String(h), tone, isInh ? "inh" : role)}</div>;
            })}
          </Row>
        </Row>
      </Stack>
      <Row gap={16} wrap>
        <span className={clsx("text-xs", inkClassName("tertiary"))}>LCP = <strong>{lcp}</strong></span>
        <span className={clsx("text-xs", inkClassName("tertiary"))}>inherited = <strong>{inherited}</strong></span>
        <span className={clsx("text-xs", inkClassName("tertiary"))}>new blocks = <strong>{newN}</strong> ({asst} asst / {newN - asst} user)</span>
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        The first <Code inline>lcp</Code> blocks whose hash ids match the parent are <strong>inherited</strong> verbatim
        (identical segment ids → cache hit). New blocks past the LCP are role-tagged:{" "}
        <Code inline>ceil(prev_out / block_size)</Code> leading blocks are the previous turn&apos;s assistant response,
        the rest are user. A turn whose new region is all-assistant flips its own last block to user, so turn
        boundaries stay block-aligned and cache-safe.
      </p>
    </Stack>
  );
}

const STORE_FILES = [
  { name: "content.blob", role: "deduplicated segment blobs" },
  { name: "content.idx", role: "packed array('Q') span pairs (interned handles)" },
  { name: "nodes.blob", role: "per-node envelopes with int handles" },
  { name: "nodes.idx", role: "trace_id → ordinal:variant → [off,size]" },
];

function UnifiedStore(): React.JSX.Element {
  return (
    <Stack gap={12}>
      <div className={clsx("rounded-none border", strokeClassName("primary"))}>
        <div className={clsx("flex items-center justify-between border-b px-4 py-2", strokeClassName("secondary"))}>
          <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>A2 interned layout</span>
          <Code inline>aiperf_graph_segments_&lt;id&gt;/</Code>
        </div>
        <div className="px-4 py-3">
          <Grid columns={2} gap={10}>
            {STORE_FILES.map((f) => (
              <div key={f.name}>
                <Code inline>{f.name}</Code>
                <div className={clsx("mt-1 text-xs", inkClassName("tertiary"))}>{f.role}</div>
              </div>
            ))}
          </Grid>
        </div>
      </div>
      <Callout tone="success">
        A2-strict: segments are addressed by insertion-index int handles, so the worker takes the zero-copy
        pre-serialized bytes path. The reader accepts ONLY the packed <Code inline>content.idx</Code>; a legacy A1
        (JSON hex) index is a pre-v3 store and is rejected loud — that run must be re-parsed.
      </Callout>
    </Stack>
  );
}

function WarmupVariant(): React.JSX.Element {
  const [warmup, setWarmup] = useState(false);
  return (
    <Stack gap={12}>
      <Row align="center" gap={10}>
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>phase_variant = {warmup ? "warmup" : "profiling"}</span>
        <div className="flex-1" />
        <span className={clsx("text-xs", inkClassName("tertiary"))}>warmup</span>
        <Toggle checked={warmup} onChange={setWarmup} />
      </Row>
      <Grid columns={2} gap={12}>
        <div className={clsx("rounded-none border", strokeClassName("secondary"))}>
          <div className={clsx("border-b px-4 py-2 text-sm font-semibold", strokeClassName("secondary"), inkClassName("primary"))}>Store lookup</div>
          <p className={clsx("px-4 py-3 text-sm", inkClassName("secondary"))}>
            Both variants read the <strong>same</strong> profiling envelope bytes:{" "}
            <Code inline>lookup = &quot;profiling&quot; if warmup else phase_variant</Code>. No duplicate warmup store.
          </p>
        </div>
        <div className={clsx("rounded-none border", strokeClassName("secondary"))}>
          <div className={clsx("border-b px-4 py-2 text-sm font-semibold", strokeClassName("secondary"), inkClassName("primary"))}>Output cap</div>
          <div className="px-4 py-3">
            <Row gap={8} align="center">
              <Code inline>max_tokens</Code>
              <span className={inkClassName("tertiary")}>=</span>
              <span className={clsx("rounded-none px-3 py-1 text-sm font-bold", warmup ? categoryBgTintClassName("orange") : categoryBgTintClassName("blue"), inkClassName("primary"))}>
                {warmup ? "1" : "512 (recorded)"}
              </span>
            </Row>
            <p className={clsx("mt-2 text-sm", inkClassName("secondary"))}>
              {warmup
                ? "Warmup forces WARMUP_MAX_OUTPUT_TOKENS (default 1): same input prefix, one-token output."
                : "Profiling uses the recorded dispatch overrides verbatim."}
            </p>
          </div>
        </div>
      </Grid>
    </Stack>
  );
}

export function DeduplicationPage({ audience }: { audience: Audience }): React.JSX.Element {
  const dev = audience === "developer";
  return (
    <Stack gap={20}>
      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Walkthrough: why prompt dedup wins</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Prompts are chopped into content-addressed segments. Toggle dedup to see how shared prefixes collapse to
          identical segment ids — the mechanism behind KV prefix-cache reuse.
        </p>
        <DedupVisual />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Content-parent prefix trie</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          How the builder decides which earlier request a new one shares a prefix with — the structure that makes dedup
          possible.
        </p>
        <ContentTrie />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>The payoff: prefix-cache token reuse</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          As a conversation grows, more of each prompt is a repeated prefix the server can serve from its KV cache.
        </p>
        <Table
          columns={[
            { key: "req", label: "Request" },
            { key: "newt", label: "New tokens", align: "end" },
            { key: "reused", label: "Reused (prefix-cache) tokens", align: "end" },
          ]}
          rows={[
            { req: "req 1", newt: "512", reused: "0" },
            { req: "req 2", newt: "128", reused: "512", tone: "success" },
            { req: "req 3", newt: "96", reused: "640", tone: "success" },
            { req: "req 4", newt: "160", reused: "736", tone: "success" },
          ]}
        />
        <p className={clsx("text-xs", inkClassName("tertiary"))}>
          Illustrative per-request prompt composition (tokens) · reused = prefix served from KV cache · source:
          segment-trie reconstruction. As the conversation grows, the reused prefix dominates.
        </p>
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Block-aligned prefix geometry</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          The builder aligns each turn to a block grid, inherits the longest common block prefix from its content
          parent, and role-tags the new blocks so turn boundaries stay cache-safe.
        </p>
        <BlockGeometry />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Block-role composition over a conversation</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          As a trie path deepens, each turn inherits more cached prefix blocks and materializes fewer new ones.
        </p>
        <Table
          columns={[
            { key: "turn", label: "Turn" },
            { key: "user", label: "New user", align: "end" },
            { key: "asst", label: "New assistant", align: "end" },
            { key: "inh", label: "Inherited (cached prefix)", align: "end" },
          ]}
          rows={[
            { turn: "turn 1", user: "4", asst: "0", inh: "0" },
            { turn: "turn 2", user: "3", asst: "2", inh: "4", tone: "success" },
            { turn: "turn 3", user: "2", asst: "2", inh: "9", tone: "success" },
            { turn: "turn 4", user: "2", asst: "3", inh: "13", tone: "success" },
          ]}
        />
        <p className={clsx("text-xs", inkClassName("tertiary"))}>
          Illustrative per-turn block composition (whole blocks) · inherited blocks reuse parent segment ids · source:
          block_role_split geometry. Inherited prefix dominates as the trie deepens.
        </p>
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Warmup vs profiling, from one envelope</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Warmup does not need its own store. It reuses the profiling bytes and only changes the output cap.
        </p>
        <WarmupVariant />
      </Stack>

      {dev && (
        <Stack gap={10}>
          <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>On-disk store shapes</h2>
          <p className={clsx("text-sm", inkClassName("secondary"))}>
            Two store shapes remain: the unified segment-trie store (content + addressing folded into one directory),
            written by every trie build, and the legacy GRAPH_DELTA store used only by native delta-mmap builds.
          </p>
          <Table
            columns={[
              { key: "store", label: "Store" },
              { key: "role", label: "Role" },
              { key: "disk", label: "On-disk" },
            ]}
            rows={[
              { store: "GraphSegmentUnifiedBackingStore", role: "content + addressing in one dir (sole trie store)", disk: <Code inline>content.&#123;blob,idx&#125; + nodes.&#123;blob,idx&#125;</Code> },
              { store: "GraphDeltaBackingStore", role: "ancestor-delta envelopes (native builds only)", disk: <Code inline>graph_deltas.dat + index.dat</Code> },
            ]}
          />
          <p className={clsx("text-xs", inkClassName("tertiary"))}>
            The unified store is A2-interned only; its packed <Code inline>content.idx</Code> enables the zero-copy
            bytes dispatch, and a pre-v3 A1 (JSON hex) store is rejected loud.
          </p>
          <UnifiedStore />
        </Stack>
      )}
    </Stack>
  );
}
