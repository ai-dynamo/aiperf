/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Credit Flow page: the life of one graph credit. Ports `CreditWalkthrough` (as a
//! `useStepSimulator`-driven participant-lane sequence), the node-kinds / dispatch-registry
//! tables, `DispatchOutcomeVisual`, `OverlapBarrierVisual`, and `BackpressureMeters` from
//! `graph-subsystem-overview.canvas.tsx`.

import { useState } from "react";
import clsx from "clsx";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { Callout } from "../../prose/Callout.js";
import { Table } from "../../prose/Table.js";
import { Code } from "../../prose/Code.js";
import { Toggle } from "../../prose/Toggle.js";
import {
  inkClassName,
  strokeClassName,
  surfaceClassName,
  categoryBgClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../../theme/tokens.js";

const PARTICIPANTS = [
  { id: "exec", label: "Executor" },
  { id: "adapter", label: "Credit adapter" },
  { id: "issuer", label: "Credit issuer" },
  { id: "worker", label: "Worker" },
  { id: "server", label: "Server" },
];

interface WalkStep {
  from: string;
  to: string;
  label: string;
  detail: string;
}

const WALK_STEPS: WalkStep[] = [
  { from: "exec", to: "adapter", label: "dispatch(node)", detail: "The executor fires an LlmNode and calls the injected credit issuer — in graph replay that issuer is the per-instance CreditDispatchAdapter." },
  { from: "adapter", to: "adapter", label: "resolve ordinal · park Future", detail: "The adapter maps the runtime node to its build-time node_ordinal, mints a correlation id, and parks an asyncio.Future keyed by (x_correlation_id, turn_index)." },
  { from: "adapter", to: "issuer", label: "issue_graph_credit(TurnToSend)", detail: "It builds a TurnToSend carrying trace_id, node_ordinal and phase_variant, then hands it to CreditIssuer.issue_graph_credit — bypassing the linear session-slot lifecycle." },
  { from: "issuer", to: "worker", label: "Credit via router", detail: "The issuer places the graph credit on the normal credit router, which delivers it to any available worker. Graph credits still take one prefill slot." },
  { from: "worker", to: "worker", label: "materialize by (trace_id, ordinal, variant)", detail: "The worker sees credit.trace_id is set, strips the #recycle suffix to the base template id, opens the shared mmap stores once, and rebuilds the exact request body by address." },
  { from: "worker", to: "server", label: "send request · stream tokens", detail: "It sends the reconstructed body through the normal InferenceClient and streams tokens back, recording per-node latency and token metrics." },
  { from: "server", to: "exec", label: "return resolves parked Future", detail: "The graph return observer routes the CreditReturn by trace_id to the live adapter, which resolves the parked Future — success, error, cancellation, or context-overflow early-exit." },
  { from: "exec", to: "exec", label: "publish outputs · schedule successors", detail: "The executor publishes the node's writes to the channel store, marks its producers done, and schedules static and selected conditional successors." },
];

function CreditWalkthrough(): React.JSX.Element {
  const sim = useStepSimulator(WALK_STEPS, { autoPlayMs: 1600 });
  const step = sim.index;
  const cur = WALK_STEPS[step]!;
  const activeIds = new Set([cur.from, cur.to]);

  return (
    <Stack gap={12}>
      <Row gap={8} wrap align="center">
        {PARTICIPANTS.map((p, i) => (
          <Row key={p.id} gap={8} align="center">
            <div
              className={clsx(
                "rounded-none border px-3 py-2 text-xs font-semibold",
                activeIds.has(p.id)
                  ? clsx(categoryBgTintClassName("blue"), strokeClassName("primary"), inkClassName("primary"))
                  : clsx(surfaceClassName("elevated"), strokeClassName("secondary"), inkClassName("secondary")),
              )}
            >
              {p.label}
            </div>
            {i < PARTICIPANTS.length - 1 && <span className={inkClassName("tertiary")}>→</span>}
          </Row>
        ))}
      </Row>

      <div className={clsx("rounded-none border px-4 py-3", strokeClassName("primary"), surfaceClassName("elevated"))}>
        <Row align="center" gap={10}>
          <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>
            Step {step + 1} / {WALK_STEPS.length}
          </span>
          <Code inline>
            {cur.from === cur.to ? `${PARTICIPANTS.find((p) => p.id === cur.from)!.label}: ${cur.label}` : cur.label}
          </Code>
          <div className="flex-1" />
          <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>Prev</Button>
          <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>Next</Button>
          <Button variant="ghost" onClick={sim.reset}>Reset</Button>
        </Row>
        <p className={clsx("mt-3 text-sm", inkClassName("primary"))}>{cur.detail}</p>
      </div>
    </Stack>
  );
}

type DispatchOutcome = "success" | "overflow" | "error" | "cancelled" | "timeout" | "refused";

const OUTCOMES: Record<DispatchOutcome, { resolves: string; effect: string; color: CategoryRole }> = {
  success: { resolves: "future.set_result(placeholder)", effect: "The node's writes publish to the channel store; static and selected conditional successors are scheduled.", color: "green" },
  overflow: { resolves: "future.set_exception(_NodeOverflowTerminate)", effect: "Context-overflow error → the trajectory terminates early (clean exit, no successors); the record still flows to the metrics-skip path.", color: "yellow" },
  error: { resolves: "future.set_exception(GraphDispatchError)", effect: "A worker error unwinds the awaiting coroutine as a normal task failure (not CancelledError), failing the trace task.", color: "red" },
  cancelled: { resolves: 'future.set_exception(GraphDispatchError("cancelled"))', effect: "The worker return reported cancellation; treated as trace failure but still advances the cross-stream barrier frontier.", color: "red" },
  timeout: { resolves: "asyncio.wait_for → TimeoutError", effect: "The dispatch_timeout_s guard fires: the orphaned waiter is popped so a late return can't resolve a dead Future, and the error re-raises.", color: "yellow" },
  refused: { resolves: "future.set_exception(GraphDispatchError)", effect: "issue_graph_credit returned False (stop/duration/request-count cap or run cancelled) — no return will ever arrive, so the parked Future is rejected immediately.", color: "blue" },
};

function DispatchOutcomeVisual(): React.JSX.Element {
  const [outcome, setOutcome] = useState<DispatchOutcome>("success");
  const r = OUTCOMES[outcome];
  return (
    <Stack gap={12}>
      <Row gap={6} wrap>
        {(Object.keys(OUTCOMES) as DispatchOutcome[]).map((o) => (
          <button
            key={o}
            type="button"
            aria-pressed={o === outcome}
            onClick={() => setOutcome(o)}
            className={clsx(
              "rounded-none border px-3 py-1 text-xs font-medium",
              strokeClassName(o === outcome ? "primary" : "secondary"),
              o === outcome ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")),
            )}
          >
            {o}
          </button>
        ))}
      </Row>
      <div className={clsx("rounded-none border", strokeClassName("primary"))}>
        <div className={clsx("flex items-center justify-between border-b px-4 py-2", strokeClassName("secondary"))}>
          <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>resolve(credit, error, cancelled)</span>
          <span className={clsx("text-xs", inkClassName("tertiary"))}>outcome: {outcome}</span>
        </div>
        <div className="flex gap-3 px-4 py-3">
          <div className={clsx("w-1 shrink-0 self-stretch", categoryBgClassName(r.color))} />
          <Stack gap={6}>
            <Code inline>{r.resolves}</Code>
            <p className={clsx("text-sm", inkClassName("secondary"))}>{r.effect}</p>
          </Stack>
        </div>
      </div>
      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        Every dispatch parks an <Code inline>asyncio.Future</Code> under a collision-free{" "}
        <Code inline>(x_correlation_id, turn_index)</Code> key. The credit-return callback routes back by{" "}
        <Code inline>credit.trace_id</Code> and calls <Code inline>resolve</Code>, which sets exactly one of these
        outcomes.
      </p>
    </Stack>
  );
}

function OverlapBarrier(): React.JSX.Element {
  const [gated, setGated] = useState(true);
  const box = (label: string, sub: string, tone: "neutral" | "park" | "issue") => (
    <div
      className={clsx(
        "rounded-none border px-3.5 py-2.5 text-center",
        strokeClassName("secondary"),
        tone === "issue" ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : tone === "park" ? clsx(categoryBgTintClassName("orange"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("primary")),
      )}
    >
      <div className="text-xs font-semibold">{label}</div>
      <div className={clsx("text-xs", inkClassName("secondary"))}>{sub}</div>
    </div>
  );
  return (
    <Stack gap={12}>
      <Row align="center" gap={10}>
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>
          {gated ? "Barrier gating dispatch order" : "Immediate issue (no barrier)"}
        </span>
        <div className="flex-1" />
        <span className={clsx("text-xs", inkClassName("tertiary"))}>overlap barrier</span>
        <Toggle checked={gated} onChange={setGated} />
      </Row>
      <Row gap={10} align="center" wrap>
        {gated ? (
          <>
            {box("node ready", "executor fires", "neutral")}
            <span className={inkClassName("tertiary")}>→</span>
            {box("submit() parks", "cross-stream pred pending", "park")}
            <span className={inkClassName("tertiary")}>→</span>
            {box("predecessor returns", "complete(ordinal)", "neutral")}
            <span className={inkClassName("tertiary")}>→</span>
            {box("released & issued", "credit on the wire", "issue")}
          </>
        ) : (
          <>
            {box("node ready", "executor fires", "neutral")}
            <span className={inkClassName("tertiary")}>→</span>
            {box("issued immediately", "credit on the wire", "issue")}
          </>
        )}
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        When a <Code inline>TraceReplayBarrier</Code> is attached, a node whose recorded cross-stream predecessors have
        not yet completed is <strong>deferred</strong> — <Code inline>submit()</Code> parks the issue closure and{" "}
        <Code inline>complete(node_ordinal)</Code> on a return releases it onto the wire. Cancel and error count as
        completion too, so a failed predecessor never deadlocks a gated sibling. With no barrier, every dispatch issues
        at once.
      </p>
    </Stack>
  );
}

const METERS: { label: string; right: string; used: number; total: number; color: CategoryRole }[] = [
  { label: "Trace lanes", right: "12 / 64 admitted", used: 12, total: 64, color: "blue" },
  { label: "Prefill slots", right: "38 / 100 in flight", used: 38, total: 100, color: "orange" },
  { label: "Adapter waiters (parked futures)", right: "7 awaiting return", used: 7, total: 24, color: "purple" },
];

function BackpressureMeters(): React.JSX.Element {
  return (
    <Stack gap={14}>
      {METERS.map((m) => (
        <div key={m.label}>
          <Row align="center" justify="space-between">
            <span className={clsx("text-sm font-medium", inkClassName("primary"))}>{m.label}</span>
            <span className={clsx("text-xs", inkClassName("tertiary"))}>{m.right}</span>
          </Row>
          <div className={clsx("mt-1 h-3 w-full rounded-none border", strokeClassName("secondary"), categoryBgTintClassName(m.color))}>
            <div className={clsx("h-full", categoryBgClassName(m.color))} style={{ width: `${(m.used / m.total) * 100}%` }} />
          </div>
        </div>
      ))}
      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        These are independent controls — do not collapse them into one number. Lanes bound admitted traces; prefill
        slots bound in-flight prefill pressure; adapter waiters are graph requests awaiting a correlated worker return.
      </p>
    </Stack>
  );
}

export function CreditFlowPage(): React.JSX.Element {
  return (
    <Stack gap={20}>
      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Walkthrough: the life of one graph credit</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Step through what happens between an executor firing an LLM node and that node completing. The highlighted
          lane shows who is acting.
        </p>
        <CreditWalkthrough />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Node kinds &amp; who issues a credit</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Every node kind is a self-registered dispatch handler. Only some actually send a request to the server under
          test; the rest sleep, replay recorded data, or run nested graphs.
        </p>
        <Table
          columns={[
            { key: "kind", label: "Node kind" },
            { key: "credit", label: "Issues credit?" },
            { key: "what", label: "What it does" },
          ]}
          rows={[
            { kind: <Code inline>llm</Code>, credit: "Always", what: "Builds a DispatchRequest and awaits the credit adapter.", tone: "success" },
            { kind: <Code inline>tool</Code>, credit: "Sometimes", what: "Bypasses credits when the endpoint provides dispatch_tool; else falls back to the issuer.", tone: "warning" },
            { kind: <Code inline>replay · delay · compact · bootstrap · tool_call · tool_result</Code>, credit: "No", what: "Optionally sleep, then write replayed or synthesized outputs." },
            { kind: <Code inline>spawn · await · subgraph · loop</Code>, credit: "No", what: "Run nested child executors (detached or inline) rather than a request." },
            { kind: <Code inline>barrier</Code>, credit: "No", what: "Synchronizes on predecessor tasks with all/any/quorum policy." },
          ]}
        />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>The dispatch handler registry</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Every node kind self-registers a handler on the executor&apos;s single-dispatch method, so the executor core
          never branches on node type.
        </p>
        <Table
          columns={[
            { key: "kinds", label: "Node kind(s)" },
            { key: "module", label: "Dispatch module" },
            { key: "credit", label: "Issues credit?" },
          ]}
          rows={[
            { kinds: <Code inline>LlmNode</Code>, module: <Code inline>dispatch/llm.py</Code>, credit: "Always", tone: "success" },
            { kinds: <Code inline>ToolNode</Code>, module: <Code inline>dispatch/tool.py</Code>, credit: "Sometimes", tone: "warning" },
            { kinds: <Code inline>BarrierNode</Code>, module: <Code inline>dispatch/barrier.py</Code>, credit: "No" },
            { kinds: <Code inline>SpawnNode</Code>, module: <Code inline>dispatch/spawn.py</Code>, credit: "No" },
            { kinds: <Code inline>AwaitNode</Code>, module: <Code inline>dispatch/await_node.py</Code>, credit: "No" },
            { kinds: <Code inline>SubgraphNode</Code>, module: <Code inline>dispatch/subgraph.py</Code>, credit: "No" },
            { kinds: <Code inline>LoopNode</Code>, module: <Code inline>dispatch/loop.py</Code>, credit: "No" },
            { kinds: <Code inline>Replay · Delay · Compact · Bootstrap · ToolResult · ToolCall</Code>, module: <Code inline>dispatch/replay.py</Code>, credit: "No" },
          ]}
        />
        <p className={clsx("text-xs", inkClassName("tertiary"))}>
          Each dispatch module self-registers its handler on the executor&apos;s <Code inline>singledispatchmethod</Code>{" "}
          via <Code inline>TraceExecutor.__dict__[&quot;_execute&quot;].register(NodeType, fn)</Code>. The six
          replay-class kinds share one body; the dispatch table stays flat and the executor never branches on node type.
        </p>
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Dispatch outcomes: resolving the parked Future</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Every LLM dispatch parks a Future and awaits a correlated worker return. Pick an outcome to see how the credit
          adapter resolves or rejects it.
        </p>
        <DispatchOutcomeVisual />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Cross-stream overlap barrier</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          An optional per-trace barrier gates dispatch order so recorded cross-stream overlap is reproduced rather than
          collapsed.
        </p>
        <OverlapBarrier />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Backpressure layers</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Graph replay has several independent throttles. Each meter bounds a different resource.
        </p>
        <BackpressureMeters />
      </Stack>

      <Callout tone="info" title="One address joins the planes">
        The credit adapter is the per-instance bridge between the async executor and the v1 credit system. It parks a
        Future per dispatched node and resolves it when the worker returns — the only coupling between scheduling and
        payload reconstruction is the stable <Code inline>(trace_id, node_ordinal, phase_variant)</Code> address.
      </Callout>
    </Stack>
  );
}
