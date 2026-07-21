/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Table } from "../../prose/Table.js";
import { Stat } from "../../prose/Stat.js";
import { Legend } from "../../prose/Legend.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

// Ported from docs/canvases/graph-fan-in.canvas.tsx. Single-view canvas — no
// internal page tabs. Source: src/aiperf/graph/channel_store.py (fan-in gate on
// a per-channel append-only log, resolved deterministically by a reducer).

const graphNodes: Node[] = [
  {
    id: "pa",
    type: "card",
    position: { x: 0, y: 0 },
    data: {
      title: "Producer A",
      detail: "write(messages) -> seq 1",
      className: "border-l-4 border-l-category-green",
    },
  },
  {
    id: "pb",
    type: "card",
    position: { x: 0, y: 100 },
    data: {
      title: "Producer B",
      detail: "write(messages) -> seq 2",
      className: "border-l-4 border-l-category-green",
    },
  },
  {
    id: "pc",
    type: "card",
    position: { x: 0, y: 200 },
    data: {
      title: "Producer C",
      detail: "skipped branch · wrote=False",
      className: "border-l-4 border-l-category-gray border-dashed",
    },
  },
  {
    id: "chan",
    type: "panel",
    position: { x: 320, y: 100 },
    data: {
      title: "channel: messages",
      detail: "reducer=add_messages · declared=3",
      className: "border-l-4 border-l-category-blue",
    },
  },
  {
    id: "gate",
    type: "panel",
    position: { x: 640, y: 100 },
    data: {
      title: "Consumer.await_inputs",
      detail: "count=2 -> wait for 2 arrivals",
      className: "border-l-4 border-l-category-purple",
    },
  },
  {
    id: "reduce",
    type: "panel",
    position: { x: 960, y: 100 },
    data: {
      title: "read + reduce",
      detail: "order by (write_seq, writer_node_id)",
      className: "border-l-4 border-l-category-orange",
    },
  },
  {
    id: "fire",
    type: "card",
    position: { x: 1280, y: 100 },
    data: {
      title: "Consumer fires",
      detail: "causal snapshot at gate seq",
      className: "border-l-4 border-l-category-green",
    },
  },
];

const graphEdges: Edge[] = [
  { id: "e-pa-chan", source: "pa", target: "chan", type: "flow" },
  { id: "e-pb-chan", source: "pb", target: "chan", type: "flow" },
  {
    id: "e-pc-chan",
    source: "pc",
    target: "chan",
    style: { strokeDasharray: "4 4" },
  },
  { id: "e-chan-gate", source: "chan", target: "gate", type: "flow" },
  { id: "e-gate-reduce", source: "gate", target: "reduce", type: "flow" },
  { id: "e-reduce-fire", source: "reduce", target: "fire", type: "flow" },
];

const LIFECYCLE_STEPS: Array<{ n: string; title: string; note: string }> = [
  {
    n: "1",
    title: "Resolve target count",
    note: 'count=N -> N; count="all" -> _producers_declared[channel] (static topology count from producers_per_channel).',
  },
  {
    n: "2",
    title: "Reachability check",
    note: "if arrivals_so_far + producers_remaining < target -> orphan now (insufficient_producers_remaining), don't sleep.",
  },
  {
    n: "3",
    title: "Register waiter, await event",
    note: "_Waiter(channel, required_count) parked on an asyncio.Event until enough arrivals land.",
  },
  {
    n: "4",
    title: "Producer writes wake waiters",
    note: "write() appends a _LogEntry(seq, writer_id, value), bumps arrival_count, sets events whose count is met.",
  },
  {
    n: "5",
    title: "Capture",
    note: "freeze the first `target` non-init writes, sorted by (write_seq, writer_node_id). Init seed (seq 0) is not an arrival.",
  },
  {
    n: "6",
    title: "Read + reduce",
    note: "reducer folds captured writes over the init seed in (write_seq, writer_node_id) order -> deterministic joined value.",
  },
];

function Header(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <Row align="center" gap={10} wrap>
        <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
          Fan-in in the graph dataflow runtime
        </h1>
        <span
          className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
        >
          VersionedChannelStore
        </span>
      </Row>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        A fan-in is one consumer whose{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>
          ChannelRequirement(channel, count)
        </span>{" "}
        reads a channel that several producer nodes write. There is no central join node — the join is a{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>count gate</span> on a per-channel
        append-only log, resolved deterministically by a reducer.
      </p>
      <Grid columns={4} gap={12}>
        <Stat value="append-log" label="Per-channel storage" />
        <Stat value="count=N | all" label="Gate modes" />
        <Stat value="(seq, writer_id)" label="Reduce order" />
        <Stat value="orphan" label="Unreachable-count exit" tone="negative" />
      </Grid>
    </Stack>
  );
}

function GraphSection(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Three producers, one count gate
      </h2>
      <Legend
        entries={[
          { color: "blue", label: "channel log" },
          { color: "purple", label: "gate" },
          { color: "gray", label: "skipped producer (wrote=False)" },
        ]}
      />
      <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
        <div
          className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}
        >
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Fan-in firing</span>
          <span
            className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
          >
            count=2 of 3
          </span>
        </div>
        <div style={{ height: 360 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={graphNodes}
            edges={graphEdges}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
      </div>
      <p className={`text-sm ${inkClassName("tertiary")}`}>
        A and B write (arrival_count reaches 2), satisfying count=2, so the consumer wakes and captures the
        first two writes. C&apos;s untaken branch is marked done with wrote=False — it decrements the
        remaining-producer count but the gate was already met. Source: src/aiperf/graph/channel_store.py.
      </p>
    </Stack>
  );
}

function Lifecycle(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Gate lifecycle — await_inputs
      </h2>
      <Stack gap={0}>
        {LIFECYCLE_STEPS.map((s, i) => (
          <div key={s.n}>
            <Row gap={12} align="start" className="py-2">
              <div
                className={`flex h-6 min-w-6.5 items-center justify-center rounded-full border text-xs font-semibold text-accent-primary`}
                style={{ borderColor: "var(--color-accent-primary)" }}
              >
                {s.n}
              </div>
              <Stack gap={2} className="min-w-0 flex-1">
                <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{s.title}</span>
                <span className={`text-sm ${inkClassName("secondary")}`}>{s.note}</span>
              </Stack>
            </Row>
            {i < LIFECYCLE_STEPS.length - 1 && <div className={`border-t ${strokeClassName("secondary")}`} />}
          </div>
        ))}
      </Stack>
    </Stack>
  );
}

function CountModes(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Gate modes</h2>
      <Table
        columns={[
          { key: "requirement", label: "Requirement" },
          { key: "satisfied", label: "Satisfied when" },
          { key: "notes", label: "Notes" },
        ]}
        rows={[
          {
            requirement: "count = N",
            satisfied: "N producer writes have arrived",
            notes: "captures first N by (write_seq, writer_id)",
          },
          {
            requirement: 'count = "all"',
            satisfied: "arrivals reach the static declared producer count",
            notes: "resolved at call time via producers_per_channel",
          },
          {
            requirement: "streaming close",
            satisfied: "STREAM_CLOSE sets arrival_count = inf",
            notes: "all waiters on that channel unblock at once",
          },
          {
            requirement: "relaxed barrier: any",
            satisfied: "one satisfied input",
            notes: "bypasses normal channel gating",
          },
          {
            requirement: "relaxed barrier: quorum",
            satisfied: "quorum_count satisfied inputs",
            notes: "must self-resolve once it can't reach quorum (F-3)",
          },
        ]}
      />
    </Stack>
  );
}

function ProducerResolution(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Producer resolution — mark_producer_done
      </h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        Every declared producer resolves exactly once. Fan-in liveness depends on producers that will never
        write telling the store so, which keeps a{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>count=&quot;all&quot;</span> waiter from
        sleeping forever.
      </p>
      <Table
        columns={[
          { key: "success", label: "success" },
          { key: "wrote", label: "wrote" },
          { key: "meaning", label: "Meaning" },
          { key: "effect", label: "Effect on the gate" },
        ]}
        rows={[
          {
            success: "True",
            wrote: "True",
            meaning: "A real write already landed",
            effect: "decrement remaining; write already bumped arrival_count",
            tone: "success",
          },
          {
            success: "True",
            wrote: "False",
            meaning: "Ran, no write (skipped conditional branch)",
            effect: "decrement only; waiters wake orphaned if count now unreachable",
            tone: "neutral",
          },
          {
            success: "False",
            wrote: "-",
            meaning: "Cancelled / failed producer",
            effect: "decrement; orphan waiters that can no longer reach count",
            tone: "warning",
          },
          {
            success: "False",
            wrote: "-",
            meaning: "Last producer, 0 arrivals, no init seed",
            effect: "channel itself is marked orphaned for future readers",
            tone: "danger",
          },
        ]}
      />
      <Grid columns={2} gap={16}>
        <Callout tone="warning" title="insufficient_producers_remaining">
          arrivals + still-live producers &lt; target. The waiter wakes with{" "}
          <span className={`font-semibold ${inkClassName("primary")}`}>ChannelOrphanedError</span> instead of
          hanging — deterministic orphan, not a deadlock.
        </Callout>
        <Callout tone="danger" title="all_producers_cancelled">
          Every producer of a required channel reported{" "}
          <span className={`font-semibold ${inkClassName("primary")}`}>success=False</span> before the count
          was met. Same forced wake, different reason string.
        </Callout>
      </Grid>
    </Stack>
  );
}

function Determinism(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Why the join is deterministic</h2>
      <Grid columns={2} gap={16}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`border-b px-4 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`}>
            Reduce order
          </div>
          <div className="px-4 py-3">
            <p className={`text-sm ${inkClassName("secondary")}`}>
              A single monotonic{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>_next_seq</span> linearizes every
              commit. Captured writes are always folded in{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>(write_seq, writer_node_id)</span>{" "}
              order, so concurrent arrivals reduce identically run-to-run regardless of which task woke first.
            </p>
          </div>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`border-b px-4 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`}>
            Init seed vs. arrival
          </div>
          <div className="px-4 py-3">
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Initial state seeds a log entry at{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>write_seq=0</span> (writer{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>__init__</span>) that is the
              reducer&apos;s starting value but does{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>not</span> count as a producer
              arrival — so a count=N gate measures node writes only.
            </p>
          </div>
        </div>
      </Grid>
      <Callout tone="info" title="Under Step/Emit (M-1)">
        The same gate becomes a Step&apos;s{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>inputs</span> fan-in. Live multi-writer
        joins (add_messages) merge in static Step-id order with FAILED writers contributing nothing;
        undeclared multi-writer live joins are rejected. weka/dynamo replay use recorded channels, so this
        only bites if the graph lane grows source=live multi-turn.
      </Callout>
    </Stack>
  );
}

/**
 * Ports `docs/canvases/graph-fan-in.canvas.tsx` (a real, hand-authored Cursor Canvas) onto
 * aiperf-flow's component vocabulary. Single-view canvas — explains fan-in in the graph
 * dataflow runtime's `VersionedChannelStore`: a count-gated per-channel append-only log
 * with a deterministic `(write_seq, writer_node_id)` reduce order, no central join node.
 */
export function GraphFanInDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Graph Fan-In" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={28}>
            <Header />
            <div className={`border-t ${strokeClassName("secondary")}`} />
            <GraphSection />
            <Lifecycle />
            <CountModes />
            <ProducerResolution />
            <Determinism />
          </Stack>
        </div>
      </div>
    </div>
  );
}
