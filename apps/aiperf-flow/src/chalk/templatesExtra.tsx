/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! An extended library of ready-made "Systems Chalk" in-card mini-diagram templates (100+), grouped
//! by domain: serialization, storage, concurrency, networking, scheduling, metrics, graph/DAG, and
//! lifecycle/errors. Each is a tiny component with sensible default labels you can override, built
//! from a handful of shape factories so the whole catalog stays consistent. Drop one into a node's
//! `data.diagram` or a `ChalkCard`'s `diagram`. All presentational; every template returns a `<Diagram>`.

import { Diagram, NodeChip, RoundNode, DbNode, DiamondNode, MiniArrow, BiArrow, MiniBars } from "./MiniDiagram.js";

const BARS = [40, 72, 100, 84];
const ROUNDS = ["1", "2", "n"] as const;

// ── shape factories ──────────────────────────────────────────────────────────────────────────────

/** `from → to` (destination accented). */
function twoStep(dFrom: string, dTo: string) {
  return function T({ from = dFrom, to = dTo }: { from?: string; to?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip>{from}</NodeChip>
        <MiniArrow />
        <NodeChip accent>{to}</NodeChip>
      </Diagram>
    );
  };
}

/** `a → b → c` (last accented). */
function threeStep(dA: string, dB: string, dC: string) {
  return function T({ a = dA, b = dB, c = dC }: { a?: string; b?: string; c?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip>{a}</NodeChip>
        <MiniArrow />
        <NodeChip>{b}</NodeChip>
        <MiniArrow />
        <NodeChip accent>{c}</NodeChip>
      </Diagram>
    );
  };
}

/** `from → ▂▄█▆` (a source emitting measured values). */
function toBars(dFrom: string) {
  return function T({ from = dFrom }: { from?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip>{from}</NodeChip>
        <MiniArrow />
        <MiniBars heights={BARS} />
      </Diagram>
    );
  };
}

/** `▂▄█▆ → to` (samples folded into an accented result). */
function fromBars(dTo: string) {
  return function T({ to = dTo }: { to?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <MiniBars heights={BARS} />
        <MiniArrow />
        <NodeChip accent>{to}</NodeChip>
      </Diagram>
    );
  };
}

/** `from → [store]` (accented cylinder). */
function toCyl(dFrom: string, dStore: string) {
  return function T({ from = dFrom, store = dStore }: { from?: string; store?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip>{from}</NodeChip>
        <MiniArrow />
        <DbNode accent>{store}</DbNode>
      </Diagram>
    );
  };
}

/** `[store] → to` (read out of a cylinder). */
function fromCyl(dStore: string, dTo: string) {
  return function T({ store = dStore, to = dTo }: { store?: string; to?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <DbNode>{store}</DbNode>
        <MiniArrow />
        <NodeChip accent>{to}</NodeChip>
      </Diagram>
    );
  };
}

/** `from → ① ② ③` (fan to numbered slots/threads). */
function toRounds(dFrom: string, rounds: readonly string[] = ROUNDS) {
  return function T({ from = dFrom }: { from?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip accent>{from}</NodeChip>
        <MiniArrow />
        {rounds.map((r, i) => (
          <RoundNode key={i}>{r}</RoundNode>
        ))}
      </Diagram>
    );
  };
}

/** `① ② ③ → to` (numbered parts merge into an accented target). */
function fromRounds(dTo: string, rounds: readonly string[] = ROUNDS) {
  return function T({ to = dTo }: { to?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        {rounds.map((r, i) => (
          <RoundNode key={i} accent={i === 1}>
            {r}
          </RoundNode>
        ))}
        <MiniArrow />
        <NodeChip accent>{to}</NodeChip>
      </Diagram>
    );
  };
}

/** `from → pred? → to` (a predicate gate). */
function gate(dFrom: string, dPred: string, dTo: string) {
  return function T({
    from = dFrom,
    pred = dPred,
    to = dTo,
  }: { from?: string; pred?: string; to?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip>{from}</NodeChip>
        <MiniArrow />
        <DiamondNode accent>{pred}</DiamondNode>
        <MiniArrow />
        <NodeChip>{to}</NodeChip>
      </Diagram>
    );
  };
}

/** `a ⇄ b` (bidirectional). */
function bi(dA: string, dB: string) {
  return function T({ a = dA, b = dB }: { a?: string; b?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip>{a}</NodeChip>
        <BiArrow />
        <NodeChip accent>{b}</NodeChip>
      </Diagram>
    );
  };
}

/** `from → [a b c]` (fan-out to several chips). */
function fanChips(dFrom: string, chips: readonly string[]) {
  return function T({ from = dFrom }: { from?: string } = {}): React.JSX.Element {
    return (
      <Diagram>
        <NodeChip accent>{from}</NodeChip>
        <MiniArrow />
        {chips.map((c, i) => (
          <NodeChip key={i}>{c}</NodeChip>
        ))}
      </Diagram>
    );
  };
}

// ── serialization / encoding ─────────────────────────────────────────────────────────────────────
export const Encode = twoStep("value", "bytes");
export const Decode = twoStep("bytes", "value");
export const Serialize = twoStep("struct", "json");
export const Deserialize = twoStep("json", "struct");
export const StrictDecode = twoStep("json", "DTO");
export const ValidateDto = gate("input", "ok?", "DTO");
export const ProjectRequest = twoStep("config", "request");
export const Envelope = twoStep("msg", "frame");
export const Unframe = twoStep("frame", "msg");
export const Parse = twoStep("text", "ast");
export const Format = twoStep("turn", "payload");
export const Tokenize = twoStep("text", "tokens");
export const Detokenize = twoStep("tokens", "text");

// ── storage ──────────────────────────────────────────────────────────────────────────────────────
export const Load = fromCyl("store", "value");
export const SaveTo = toCyl("value", "store");
export const Intern = toCyl("value", "pool");
export const Snapshot = toCyl("state", "snap");
export const Checkpoint = toCyl("state", "ckpt");
export const IndexInto = toCyl("key", "index");
export const Dedup = twoStep("items", "uniq");
export const Compact = twoStep("segs", "seg");
export const AppendLog = toCyl("rec", "log");
export const ReadLog = fromCyl("log", "rec");
export const HashOf = twoStep("data", "digest");
export const PrefixOf = twoStep("tokens", "prefix");
export const ToHandle = twoStep("bytes", "Handle");
export const FromHandle = twoStep("Handle", "bytes");
export const Persist = toCyl("record", "disk");
export const Blob = toCyl("media", "blob");
export const Segmentize = fanChips("graph", ["seg", "seg", "seg"]);

// ── concurrency ──────────────────────────────────────────────────────────────────────────────────
export const JoinThreads = fromRounds("result");
export const Channel = twoStep("tx", "rx");
export const Mpsc = fromRounds("rx");
export const Barrier = fromRounds("sync");
export const Actor = twoStep("msg", "actor");
export const Mailbox = toCyl("msg", "queue");
export const ArcClone = toRounds("Arc");
export const MutexGuard = fromCyl("data", "lock");
export const AtomicRmw = twoStep("load", "store");
export const SpawnLocal = twoStep("task", "LocalSet");
export const YieldNow = twoStep("task", "loop");
export const AwaitFut = twoStep("fut", "val");
export const SelectReady = fanChips("select", ["fut", "fut", "fut"]);
export const ThreadPool = toRounds("pool");
export const Fork = fanChips("fork", ["a", "b"]);
export const SendSync = bi("Send", "Sync");

// ── networking ───────────────────────────────────────────────────────────────────────────────────
export const Request = twoStep("client", "server");
export const Response = twoStep("server", "client");
export const RoundTrip = bi("client", "server");
export const StreamOut = twoStep("src", "stream");
export const Sse = twoStep("server", "t₁·t₂·t₃");
export const Unary = twoStep("client", "server");
export const Connect = twoStep("client", "server");
export const ConnectRetry = gate("client", "↻", "server");
export const Backoff = gate("send", "↻", "ack");
export const TlsHandshake = bi("client", "server");
export const Uds = threeStep("client", "sock", "server");
export const ProxyHop = threeStep("client", "proxy", "server");
export const LoadBalance = fanChips("req", ["a", "b", "c"]);
export const Ingress = twoStep("client", "edge");
export const Egress = twoStep("exec", "net");
export const Publish = toRounds("pub");
export const Subscribe = fromRounds("sub");
export const Http = twoStep("hyper", "t₁·t₂·t₃");
export const Grpc = twoStep("Tonic", "unary");

// ── scheduling ───────────────────────────────────────────────────────────────────────────────────
export const Admit = gate("req", "slot?", "run");
export const RateLimit = gate("req", "rate?", "out");
export const TokenBucket = gate("req", "bucket", "out");
export const Backpressure = gate("fast", "gate", "slow");
export const PriorityQueue = fromRounds("out");
export const Drain = twoStep("inflight", "∅");
export const Warmup = twoStep("cold", "warm");
export const PhaseRun = threeStep("warmup", "profile", "drain");
export const Pace = twoStep("clock", "emit");
export const Arrival = twoStep("poisson", "emit");
export const ConcurrencyGate = toRounds("gate");
export const Grace = threeStep("inflight", "grace", "done");
export const CancelTask = twoStep("task", "∅");
export const TimeoutReq = gate("req", "⏱", "∅");
export const Schedule = twoStep("at_ns", "wake");
export const Dispatch = twoStep("req", "sink");

// ── metrics ──────────────────────────────────────────────────────────────────────────────────────
export const Histogram = toBars("values");
export const Percentile = fromBars("p99");
export const Counter = twoStep("events", "count");
export const Gauge = twoStep("sample", "gauge");
export const RatePerSec = twoStep("events", "per-s");
export const SumOf = fromBars("sum");
export const TDigest = toCyl("values", "t-digest");
export const SketchMerge = fromRounds("sketch");
export const SweepLine = toBars("inflight");
export const WindowOver = twoStep("stream", "window");
export const AccumulateInto = toCyl("records", "store");
export const FoldBars = fromBars("acc");
export const Extrema = fromBars("min·max");
export const StdDev = fromBars("σ");

// ── graph / DAG ──────────────────────────────────────────────────────────────────────────────────
export const DagStep = twoStep("node", "node");
export const TopoOrder = fromRounds("order");
export const ReplayTrace = twoStep("trace", "graph");
export const Trajectory = twoStep("graph", "path");
export const EmitOutputs = fanChips("node", ["out", "out"]);
export const SampleDist = twoStep("dist", "sample");
export const GraphSink = twoStep("node", "sink");
export const CountGate = gate("writes", "=N?", "wake");
export const SpanDedup = twoStep("spans", "uniq");
export const LoopBody = gate("body", "↻k", "done");
export const FanJoin = fromRounds("join");
export const Warmpath = threeStep("resolve", "compile", "execute");

// ── lifecycle / errors ───────────────────────────────────────────────────────────────────────────
export const Init = twoStep("cfg", "app");
export const Bootstrap = twoStep("app", "ready");
export const CommitParts = fromRounds("commit");
export const Rollback = twoStep("tx", "∅");
export const Fallback = twoStep("primary", "fallback");
export const CircuitBreaker = gate("req", "open?", "trip");
export const ForceKill = twoStep("inflight", "kill");
export const Terminal = twoStep("run", "done");
export const Prepare = twoStep("plan", "ready");
export const Finalize = fromBars("report");
export const Escalate = threeStep("grace", "cancel", "force");
export const Settle = twoStep("pending", "settled");
export const SelfExec = bi("parent", "--execute");
export const ReExec = twoStep("aiperf", "--execute");
