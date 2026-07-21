/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Overview page of the AIPerf Graph Subsystem deck. Ports the `StageDetail`, `CoreIdea`,
//! adapter table + `AdapterDetectionDemo`, `OrdinalAgreementVisual`, and the "what makes it
//! fast" cards from `graph-subsystem-overview.canvas.tsx`. The hand-drawn SVG `FlowDiagram`
//! becomes a real React Flow node/edge graph; the manager/developer split is preserved.

import { useMemo, useState } from "react";
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
import { Stat } from "../../prose/Stat.js";
import { Code } from "../../prose/Code.js";
import { Select } from "../../prose/Select.js";
import { Divider } from "../../layout/Divider.js";
import {
  inkClassName,
  strokeClassName,
  surfaceClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../../theme/tokens.js";
import clsx from "clsx";
import type { Audience } from "./audience.js";

type StageId = "ingest" | "build" | "schedule" | "execute" | "materialize";

interface Stage {
  id: StageId;
  label: string;
  plane: string;
  what: string;
  why: string;
  symbols: string[];
  files: string[];
  color: CategoryRole;
}

const STAGES: Stage[] = [
  {
    id: "ingest",
    label: "1 · Ingest",
    plane: "Build plane",
    color: "blue",
    what: "A recorded agent trace (Weka KV-tester or Dynamo) or a hand-authored native graph is parsed into a ParsedGraph — one LLM node per recorded request, with timing edges derived from the recorded clock.",
    why: "Every third-party trace format collapses into one canonical shape. The rest of the system never has to know where the workload came from — add an adapter, and the whole runtime comes for free.",
    symbols: ["GraphAdapterProtocol", "from_weka_trace", "from_dynamo_trace", "parse_graph_workload"],
    files: ["protocols.py", "workload_detect.py", "weka/trace.py", "dynamo/trace.py"],
  },
  {
    id: "build",
    label: "2 · Build stores",
    plane: "Build plane",
    color: "purple",
    what: "Prompts are chopped into content-addressed segments (blake2b, chained root-to-tip) and written to memory-mapped stores. Each node gets a small envelope addressed by (trace_id, node_ordinal).",
    why: "Shared prompt prefixes produce identical segment ids, so the workload reproduces real KV prefix-cache hits on the server under test. Deduplication keeps a huge corpus in bounded memory.",
    symbols: ["build_trie_ir", "SegmentPool", "segment_id", "trie_node_ordinals", "build_unified_trie_store_interned"],
    files: ["segment_ir/trie_content.py", "segment_ir/pool.py", "segment_ir/store_builder.py", "dataset_manager.py"],
  },
  {
    id: "schedule",
    label: "3 · Schedule",
    plane: "Schedule plane",
    color: "cyan",
    what: "The timing manager re-derives the identical graph (or reads a graph_meta sidecar), then admits traces into concurrent lanes, samples a warmup/profiling split point (t*), and stamps each request as warmup or profiling.",
    why: "Build and schedule are separate processes that must agree on node ordinals. Re-deriving the same parse guarantees a worker can rebuild any node's request without shared state.",
    symbols: ["GraphIRReplayStrategy", "phase_variant", "chop_trie_at_tstar", "CreditDispatchAdapter"],
    files: ["graph_ir_replay.py", "timing/manager.py", "graph_ir_source.py"],
  },
  {
    id: "execute",
    label: "4 · Execute",
    plane: "Runtime plane",
    color: "orange",
    what: "A per-trace async dataflow executor fires each node the moment its channel inputs and recorded timing gates clear. There is no central queue — readiness is expressed through channel waiters and task creation.",
    why: "Concurrency comes from two independent dials at once: many trace lanes, and many ready nodes inside a single trace. Fan-out in the original workload is preserved rather than serialized.",
    symbols: ["TraceExecutor", "VersionedChannelStore", "Scheduler", "dispatch/*", "count=all"],
    files: ["graph/executor.py", "graph/channel_store.py", "graph/scheduler.py", "credit_dispatch_adapter.py"],
  },
  {
    id: "materialize",
    label: "5 · Materialize",
    plane: "Worker plane",
    color: "green",
    what: "A graph credit routes to any worker. The worker strips the recycle suffix, opens the shared mmap stores, and rebuilds the exact request body by address — often streaming pre-serialized bytes with zero re-encoding.",
    why: "Because the body is rebuilt from shared stores, any worker can serve any node — no session affinity. Warmup reuses the profiling bytes and caps output to one token, so no duplicate stores are needed.",
    symbols: ["materialize_graph_request", "materialize_graph_request_unified_bytes", "GraphSegmentUnifiedClient", "_interned"],
    files: ["workers/worker.py", "worker_materialize.py", "graph_segment_unified_store.py"],
  },
];

const STAGE_COLOR: Record<StageId, CategoryRole> = {
  ingest: "blue",
  build: "purple",
  schedule: "cyan",
  execute: "orange",
  materialize: "green",
};

interface FlowNodeMeta {
  id: string;
  label: string;
  sub: string;
  stage: StageId;
  x: number;
  y: number;
}

const FLOW_NODES: FlowNodeMeta[] = [
  { id: "trace", label: "Recorded trace", sub: "weka · dynamo · native", stage: "ingest", x: 0, y: 40 },
  { id: "adapter", label: "Graph adapter", sub: "→ ParsedGraph", stage: "ingest", x: 210, y: 40 },
  { id: "ir", label: "Segment-trie IR", sub: "dedup prompts", stage: "build", x: 420, y: 40 },
  { id: "stores", label: "mmap stores", sub: "content + address", stage: "build", x: 420, y: 180 },
  { id: "sched", label: "Schedule", sub: "lanes · t*", stage: "schedule", x: 630, y: 40 },
  { id: "exec", label: "Executor", sub: "async dataflow", stage: "execute", x: 840, y: 40 },
  { id: "worker", label: "Worker", sub: "rebuild + send", stage: "materialize", x: 1050, y: 110 },
  { id: "server", label: "Server", sub: "under test", stage: "materialize", x: 1260, y: 110 },
];

const FLOW_EDGES: Edge[] = [
  { id: "e-trace-adapter", source: "trace", target: "adapter", type: "flow" },
  { id: "e-adapter-ir", source: "adapter", target: "ir", type: "flow" },
  { id: "e-ir-sched", source: "ir", target: "sched", type: "flow" },
  { id: "e-sched-exec", source: "sched", target: "exec", type: "flow" },
  { id: "e-exec-worker", source: "exec", target: "worker", type: "flow" },
  { id: "e-worker-server", source: "worker", target: "server", type: "flow" },
  { id: "e-ir-stores", source: "ir", target: "stores", type: "flow" },
  { id: "e-stores-worker", source: "stores", target: "worker", type: "flow", data: { speed: "slow" } },
  { id: "e-worker-exec", source: "worker", target: "exec", type: "flow", data: { color: "var(--color-ink-tertiary)", speed: "slow" } },
];

const BEATS = [
  { n: "1", t: "Canonicalize", d: "Any recorded agentic trace becomes one ParsedGraph: an LLM node per request, edges from finished-before order." },
  { n: "2", t: "Deduplicate", d: "Prompts split into content-addressed segments. Shared prefixes → identical ids → real prefix-cache hits." },
  { n: "3", t: "Replay as dataflow", d: "Nodes fire when inputs + recorded timing clear. Concurrency = trace lanes × ready nodes per trace." },
  { n: "4", t: "Rebuild anywhere", d: "A credit routes to any worker, which reconstructs the exact request by address — no shared state." },
];

const DETECT_CASES: Record<string, { winner: string | null; note: string }> = {
  gz: { winner: "dynamo_trace", note: "Directory of *.NNNNNN.jsonl.gz matches the dynamo sniff (priority 100)." },
  weka: { winner: "weka_trace", note: "A .json file / dir whose signature keys match Weka (priority 85)." },
  hf: { winner: "weka_trace", note: "An org/name id containing 'weka' with no file suffix resolves to the Weka HF path." },
  yaml: { winner: null, note: "A plain .yaml is NOT auto-detected as graph — it could be an ordinary dataset. Pass --graph-format native to force it." },
  native: { winner: "native", note: "With --graph-format native the file is forced into the native adapter (priority 1)." },
};

const DETECT_ADAPTERS = [
  { id: "dynamo_trace", pr: 100 },
  { id: "weka_trace", pr: 85 },
  { id: "native", pr: 1 },
];

function StageFlow({
  selected,
  onSelect,
}: {
  selected: StageId;
  onSelect: (s: StageId) => void;
}): React.JSX.Element {
  const nodes: Node[] = useMemo(
    () =>
      FLOW_NODES.map((n) => ({
        id: n.id,
        type: "card",
        position: { x: n.x, y: n.y },
        data: {
          title: n.label,
          subtitle: n.sub,
          className: clsx(
            categoryBgTintClassName(STAGE_COLOR[n.stage]),
            n.stage === selected && "border-l-4",
          ),
          strokeRole: n.stage === selected ? "primary" : "secondary",
        },
      })),
    [selected],
  );

  return (
    <div style={{ height: 340 }}>
      <ReactFlow
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={nodes}
        edges={FLOW_EDGES}
        onNodeClick={(_, node) => {
          const meta = FLOW_NODES.find((n) => n.id === node.id);
          if (meta) onSelect(meta.stage);
        }}
        fitView
        fitViewOptions={{ padding: 0.12 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
      </ReactFlow>
    </div>
  );
}

export function OverviewPage({ audience }: { audience: Audience }): React.JSX.Element {
  const dev = audience === "developer";
  const [stage, setStage] = useState<StageId>("ingest");
  const [detect, setDetect] = useState<string>("gz");
  const active = STAGES.find((s) => s.id === stage)!;
  const detectCase = DETECT_CASES[detect]!;

  return (
    <Stack gap={20}>
      <div>
        <h1 className={clsx("text-2xl font-bold", inkClassName("primary"))}>AIPerf Graph Subsystem</h1>
        <p className={clsx("mt-1 text-sm", inkClassName("secondary"))}>
          How recorded agentic workloads are replayed against an inference server — build once, replay anywhere.
        </p>
      </div>

      <Grid columns={4} gap={16}>
        <Stat value="3" label="Trace formats in" />
        <Stat value="1" label="Shared segment-trie IR" />
        <Stat value="4" label="Decoupled planes" />
        <Stat value="N" label="Stateless workers" />
      </Grid>

      <Callout tone="info" title="The one big idea">
        Scheduling and payload reconstruction are kept in <strong>separate planes</strong> that share no mutable
        state. They are joined only by a stable address — <Code inline>(trace_id, node_ordinal, phase_variant)</Code>{" "}
        — so an offline build can hand faithful, prefix-cache-preserving requests to any worker at replay time.
      </Callout>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>The concept in four beats</h2>
        <Grid columns={4} gap={12}>
          {BEATS.map((b) => (
            <div key={b.n} className={clsx("rounded-none border px-4 py-3", surfaceClassName("elevated"), strokeClassName("secondary"))}>
              <div className={clsx("text-sm font-bold", inkClassName("secondary"))}>{b.n}</div>
              <div className={clsx("mt-1 text-sm font-semibold", inkClassName("primary"))}>{b.t}</div>
              <p className={clsx("mt-1 text-xs", inkClassName("secondary"))}>{b.d}</p>
            </div>
          ))}
        </Grid>
      </Stack>

      <Stack gap={10}>
        <Row align="center" justify="space-between">
          <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>The pipeline, stage by stage</h2>
          <span className={clsx("text-xs", inkClassName("tertiary"))}>
            {dev ? "Developer view — files & symbols shown" : "Manager view — concepts only"}
          </span>
        </Row>
        <Row gap={8} wrap>
          {STAGES.map((s) => (
            <button
              key={s.id}
              type="button"
              aria-pressed={s.id === stage}
              onClick={() => setStage(s.id)}
              className={clsx(
                "rounded-none border px-3 py-1 text-xs font-medium",
                strokeClassName(s.id === stage ? "primary" : "secondary"),
                s.id === stage ? clsx(categoryBgTintClassName(s.color), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")),
              )}
            >
              {s.label}
            </button>
          ))}
        </Row>

        <StageFlow selected={stage} onSelect={setStage} />
        <p className={clsx("text-xs", inkClassName("tertiary"))}>
          End-to-end flow across four planes · the slow return edge = LLM return unblocks the executor · click any node
          to focus its stage.
        </p>

        <div className={clsx("rounded-none border", strokeClassName("primary"))}>
          <div className={clsx("flex items-center justify-between border-b px-4 py-2", strokeClassName("secondary"))}>
            <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{active.label.replace(/^\d+ · /, "")}</span>
            <span className={clsx("text-xs", inkClassName("tertiary"))}>{active.plane}</span>
          </div>
          <div className="px-4 py-3">
            <Stack gap={10}>
              <div>
                <div className={clsx("text-xs font-semibold uppercase", inkClassName("tertiary"))}>What happens</div>
                <p className={clsx("mt-1 text-sm", inkClassName("primary"))}>{active.what}</p>
              </div>
              <div>
                <div className={clsx("text-xs font-semibold uppercase", inkClassName("tertiary"))}>Why it matters</div>
                <p className={clsx("mt-1 text-sm", inkClassName("secondary"))}>{active.why}</p>
              </div>
              {dev && (
                <>
                  <Divider />
                  <div>
                    <div className={clsx("mb-1 text-xs font-semibold uppercase", inkClassName("tertiary"))}>Key symbols</div>
                    <Row gap={6} wrap>
                      {active.symbols.map((sym) => (
                        <Code key={sym} inline>{sym}</Code>
                      ))}
                    </Row>
                  </div>
                  <div>
                    <div className={clsx("mb-1 text-xs font-semibold uppercase", inkClassName("tertiary"))}>Open source files</div>
                    <Row gap={6} wrap>
                      {active.files.map((f) => (
                        <Code key={f} inline>{f}</Code>
                      ))}
                    </Row>
                  </div>
                </>
              )}
            </Stack>
          </div>
        </div>
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>One runtime, many recorded formats</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Adapters are plugins chosen by <Code inline>--graph-format</Code> or by priority-ordered auto-detection.
          Everything funnels into the same segment-trie IR and the same executor.
        </p>
        <Table
          columns={[
            { key: "adapter", label: "Adapter" },
            { key: "priority", label: "Priority", align: "end" },
            { key: "input", label: "Input" },
            { key: "entry", label: "Entry point" },
          ]}
          rows={[
            { adapter: "dynamo_trace", priority: "100", input: "agent-trace v1 .jsonl.gz dir", entry: <Code inline>from_dynamo_trace</Code> },
            { adapter: "weka_trace", priority: "85", input: "KV-tester JSON / dir / HF corpus", entry: <Code inline>from_weka_trace</Code> },
            { adapter: "native", priority: "1", input: "hand-authored YAML / JSONL", entry: <Code inline>parse_native</Code> },
          ]}
        />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Adapter auto-detection</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Without an explicit <Code inline>--graph-format</Code>, the highest-priority adapter whose sniff matches wins
          — and native is deliberately excluded so plain files aren&apos;t hijacked.
        </p>
        <Row gap={10} align="center" wrap>
          <span className={clsx("text-xs", inkClassName("tertiary"))}>Input</span>
          <Select
            value={detect}
            onChange={setDetect}
            options={[
              { value: "gz", label: "dir of *.jsonl.gz" },
              { value: "weka", label: "weka .json / dir" },
              { value: "hf", label: "org/name (…weka…)" },
              { value: "yaml", label: ".yaml (no flag)" },
              { value: "native", label: ".yaml + --graph-format native" },
            ]}
          />
        </Row>
        <Grid columns={3} gap={12}>
          {DETECT_ADAPTERS.map((a) => {
            const on = a.id === detectCase.winner;
            return (
              <div
                key={a.id}
                className={clsx(
                  "rounded-none border px-4 py-3",
                  strokeClassName(on ? "primary" : "secondary"),
                  on ? categoryBgTintClassName("blue") : surfaceClassName("elevated"),
                )}
              >
                <Row align="center" justify="space-between">
                  <Code inline>{a.id}</Code>
                  <span className={clsx("text-xs", inkClassName("tertiary"))}>priority {a.pr}</span>
                </Row>
                {on && <div className={clsx("mt-1 text-xs font-semibold", inkClassName("primary"))}>wins</div>}
              </div>
            );
          })}
        </Grid>
        <Callout tone={detectCase.winner ? "info" : "warning"}>{detectCase.note}</Callout>
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Why build and schedule agree</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          The build and schedule planes run in different processes yet must produce the same node ordinals. Here is the
          contract that keeps them in lockstep.
        </p>
        <Grid columns={2} gap={12}>
          {[
            { title: "DatasetManager", sub: "build proc" },
            { title: "TimingManager", sub: "schedule proc" },
          ].map((c) => (
            <div key={c.title} className={clsx("rounded-none border", strokeClassName("secondary"))}>
              <div className={clsx("flex items-center justify-between border-b px-4 py-2", strokeClassName("secondary"))}>
                <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{c.title}</span>
                <span className={clsx("text-xs", inkClassName("tertiary"))}>{c.sub}</span>
              </div>
              <div className="px-4 py-3">
                <Stack gap={4}>
                  <Code inline>parse_graph_workload(run, path)</Code>
                  <span className={clsx("text-xs", inkClassName("tertiary"))}>↓</span>
                  <Code inline>trie_node_ordinals(...)</Code>
                  <span className={clsx("text-xs", inkClassName("tertiary"))}>↓</span>
                  <div className={clsx("rounded-none px-3 py-1.5 text-sm", surfaceClassName("panel"), inkClassName("primary"))}>
                    t-1 → {"{"} n0, n1, n2 {"}"}
                  </div>
                </Stack>
              </div>
            </div>
          ))}
        </Grid>
        <Callout tone="success" title="Same parse → same ordinals">
          Two separate processes derive the identical dense ordinals from the same run-derived knobs. That shared
          address — <Code inline>(trace_id, node_ordinal, phase_variant)</Code> — is the only thing a worker needs to
          rebuild any node&apos;s request, with no shared runtime state.
        </Callout>
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>What makes it fast (and faithful)</h2>
        <Grid columns={dev ? 2 : 3} gap={12}>
          {[
            { t: "Prefix-cache fidelity", d: "Two requests that share a block-aligned prefix render an identical leading message chain — a build-time guarantee, provable offline." },
            { t: "Reconstructed timing", d: "Edges come from a finished-before frontier over the recorded clock, with idle-gap warping to cap dead air. Concurrency is preserved, not invented." },
            { t: "Two concurrency dials", d: "Cross-trace lanes and in-trace ready-node fan-out are independent knobs. A fan-out-heavy graph needs both sized." },
            ...(dev ? [{ t: "Zero-copy dispatch", d: "When cache-busting is off, workers send pre-serialized request bytes straight from mmap slices — no decode/re-encode of the messages array." }] : []),
          ].map((c) => (
            <div key={c.t} className={clsx("rounded-none border px-4 py-3", surfaceClassName("elevated"), strokeClassName("secondary"))}>
              <div className={clsx("text-sm font-semibold", inkClassName("primary"))}>{c.t}</div>
              <p className={clsx("mt-1 text-xs", inkClassName("secondary"))}>{c.d}</p>
            </div>
          ))}
        </Grid>
      </Stack>
    </Stack>
  );
}
