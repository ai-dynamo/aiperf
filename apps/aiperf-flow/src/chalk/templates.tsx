/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ready-made "Systems Chalk" in-card mini-diagram TEMPLATES — parameterized compositions of the
//! `MiniDiagram` atoms for the patterns a card most often needs (a pipeline, a store lookup, a token
//! stream, a fan-out, a metrics emit, a two-trait seam, a handshake, …). Drop one into a node's
//! `data.diagram` or a `ChalkCard`'s `diagram` prop and pass short domain labels; the card's accent
//! colors the emphasized atom. Purely presentational; every template returns a `<Diagram>`.

import {
  Diagram,
  NodeChip,
  RoundNode,
  DbNode,
  DiamondNode,
  MiniArrow,
  BiArrow,
  MiniBars,
} from "./MiniDiagram.js";

const BARS = [40, 72, 100, 84];

/** A chain of 2–4 labeled chips joined by arrows. `accentIndex` colors one step (default: last). */
export function Pipeline({
  steps,
  accentIndex = steps.length - 1,
}: {
  steps: readonly string[];
  accentIndex?: number;
}): React.JSX.Element {
  return (
    <Diagram>
      {steps.map((s, i) => (
        <span key={i} className="contents">
          {i > 0 && <MiniArrow />}
          <NodeChip accent={i === accentIndex}>{s}</NodeChip>
        </span>
      ))}
    </Diagram>
  );
}

/** `from → to`, the destination accented — a plain transformation/handoff. */
export function Transform({ from, to }: { from: string; to: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{from}</NodeChip>
      <MiniArrow />
      <NodeChip accent>{to}</NodeChip>
    </Diagram>
  );
}

/** `key → [store]` — write/intern into a store or arena (accented cylinder). */
export function Store({ key, store = "store" }: { key: string; store?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{key}</NodeChip>
      <MiniArrow />
      <DbNode accent>{store}</DbNode>
    </Diagram>
  );
}

/** `key → [cache]` — a lookup against a KV / cache / segment store. */
export function Lookup({ key = "key", store = "KV" }: { key?: string; store?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{key}</NodeChip>
      <MiniArrow />
      <DbNode accent>{store}</DbNode>
    </Diagram>
  );
}

/** `key → hit? → [store]` — a cache probe with a hit/miss branch. */
export function CacheProbe({ key = "hash", store = "KV" }: { key?: string; store?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{key}</NodeChip>
      <MiniArrow />
      <DiamondNode accent>hit?</DiamondNode>
      <MiniArrow />
      <DbNode>{store}</DbNode>
    </Diagram>
  );
}

/** `from → t₁·t₂·t₃` — a token stream out of a source. */
export function TokenStream({ from = "GPU" }: { from?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip accent>{from}</NodeChip>
      <MiniArrow />
      <NodeChip>t₁·t₂·t₃</NodeChip>
    </Diagram>
  );
}

/** `from → ▂▄█▆` — a source emitting measured values. */
export function Metrics({ from, heights = BARS }: { from: string; heights?: readonly number[] }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{from}</NodeChip>
      <MiniArrow />
      <MiniBars heights={heights} />
    </Diagram>
  );
}

/** `▂▄█▆ → to` — samples folded/emitted into a destination (accented). */
export function Emit({ to, heights = BARS }: { to: string; heights?: readonly number[] }): React.JSX.Element {
  return (
    <Diagram>
      <MiniBars heights={heights} />
      <MiniArrow />
      <NodeChip accent>{to}</NodeChip>
    </Diagram>
  );
}

/** `▂▄█▆ → to` — a reduce/fold of a stream into one accented result. Alias of `Emit` with intent. */
export function Reduce({ to, heights = BARS }: { to: string; heights?: readonly number[] }): React.JSX.Element {
  return <Emit to={to} heights={heights} />;
}

/** `① ② ③ → to` — a queue / slot pool of numbered slots feeding a target. */
export function Queue({
  slots = ["1", "2", "n"],
  to,
  activeSlot = 1,
}: {
  slots?: readonly string[];
  to: string;
  activeSlot?: number;
}): React.JSX.Element {
  return (
    <Diagram>
      {slots.map((s, i) => (
        <RoundNode key={i} accent={i === activeSlot}>
          {s}
        </RoundNode>
      ))}
      <MiniArrow />
      <NodeChip>{to}</NodeChip>
    </Diagram>
  );
}

/** `① ② ③ → to` — worker threads / sub-cells running a target. */
export function Workers({
  threads = ["w₀", "w₁", "wₙ"],
  to,
}: {
  threads?: readonly string[];
  to: string;
}): React.JSX.Element {
  return (
    <Diagram>
      {threads.map((t, i) => (
        <RoundNode key={i} accent={i === 1}>
          {t}
        </RoundNode>
      ))}
      <MiniArrow />
      <NodeChip accent>{to}</NodeChip>
    </Diagram>
  );
}

/** `from → ① ② ③` — a parent spawning child tasks/threads. */
export function Spawn({
  from,
  children = ["c₀", "c₁", "cₙ"],
}: {
  from: string;
  children?: readonly string[];
}): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip accent>{from}</NodeChip>
      <MiniArrow />
      {children.map((c, i) => (
        <RoundNode key={i}>{c}</RoundNode>
      ))}
    </Diagram>
  );
}

/** `① ② ③ → to` — partial results merged into one accented commit. */
export function Merge({
  parts = ["p₀", "p₁", "pₙ"],
  to,
}: {
  parts?: readonly string[];
  to: string;
}): React.JSX.Element {
  return (
    <Diagram>
      {parts.map((p, i) => (
        <RoundNode key={i}>{p}</RoundNode>
      ))}
      <MiniArrow />
      <NodeChip accent>{to}</NodeChip>
    </Diagram>
  );
}

/** `from → [a b c]` — one source fanning out to several targets. */
export function FanOut({ from, to }: { from: string; to: readonly string[] }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip accent>{from}</NodeChip>
      <MiniArrow />
      {to.map((t, i) => (
        <NodeChip key={i}>{t}</NodeChip>
      ))}
    </Diagram>
  );
}

/** `[a b c] → to` — several sources fanning in to one accented target. */
export function FanIn({ from, to }: { from: readonly string[]; to: string }): React.JSX.Element {
  return (
    <Diagram>
      {from.map((f, i) => (
        <NodeChip key={i}>{f}</NodeChip>
      ))}
      <MiniArrow />
      <NodeChip accent>{to}</NodeChip>
    </Diagram>
  );
}

/** `from → pred? → to` — a predicate gate on a path. */
export function Branch({
  from,
  pred = "?",
  to,
}: {
  from: string;
  pred?: string;
  to: string;
}): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{from}</NodeChip>
      <MiniArrow />
      <DiamondNode accent>{pred}</DiamondNode>
      <MiniArrow />
      <NodeChip>{to}</NodeChip>
    </Diagram>
  );
}

/** `a + b` — a seam of exactly two traits/contracts side by side. */
export function TwoTrait({ a, b }: { a: string; b: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip accent>{a}</NodeChip>
      <span className="font-mono text-[10px] text-ink-tertiary">+</span>
      <NodeChip accent>{b}</NodeChip>
    </Diagram>
  );
}

/** `trait → impl` — dynamic dispatch from a trait object to a concrete impl. */
export function TraitImpl({ trait, impl }: { trait: string; impl: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{trait}</NodeChip>
      <MiniArrow />
      <NodeChip accent>{impl}</NodeChip>
    </Diagram>
  );
}

/** `a ⇄ b` — a bidirectional handshake (stdio, request/response, sync). */
export function Handshake({ a, b }: { a: string; b: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{a}</NodeChip>
      <BiArrow />
      <NodeChip accent>{b}</NodeChip>
    </Diagram>
  );
}

/** `from → SINK` — a request driven into a (worker) sink. */
export function Sink({ from, sink = "SINK" }: { from: string; sink?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{from}</NodeChip>
      <MiniArrow />
      <NodeChip accent>{sink}</NodeChip>
    </Diagram>
  );
}

/** `from → JSON·CSV` — a typed report / artifact write-out. */
export function Report({ from = "REPORT", to = "JSON·CSV" }: { from?: string; to?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip accent>{from}</NodeChip>
      <MiniArrow />
      <NodeChip>{to}</NodeChip>
    </Diagram>
  );
}

/** `Clock → now_ns` — a clock reading / time source. */
export function Clock({ read = "now_ns" }: { read?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>Clock</NodeChip>
      <MiniArrow />
      <NodeChip accent>{read}</NodeChip>
    </Diagram>
  );
}

/** A single frozen/shared store (accented cylinder), optionally fed from a source. */
export function Frozen({ store = "store", from }: { store?: string; from?: string }): React.JSX.Element {
  return (
    <Diagram>
      {from !== undefined && (
        <>
          <NodeChip>{from}</NodeChip>
          <MiniArrow />
        </>
      )}
      <DbNode accent>{store}</DbNode>
    </Diagram>
  );
}

/** `from → ↻n → to` — a retry/backoff loop on a path. */
export function Retry({ from, to, times = "↻n" }: { from: string; to: string; times?: string }): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip>{from}</NodeChip>
      <MiniArrow />
      <DiamondNode accent>{times}</DiamondNode>
      <MiniArrow />
      <NodeChip>{to}</NodeChip>
    </Diagram>
  );
}

/** `from → ① ② ③` — broadcast/notify to several observers. */
export function Broadcast({
  from,
  to = ["o₀", "o₁", "oₙ"],
}: {
  from: string;
  to?: readonly string[];
}): React.JSX.Element {
  return (
    <Diagram>
      <NodeChip accent>{from}</NodeChip>
      <MiniArrow />
      {to.map((t, i) => (
        <RoundNode key={i}>{t}</RoundNode>
      ))}
    </Diagram>
  );
}
