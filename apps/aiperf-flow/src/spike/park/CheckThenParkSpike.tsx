/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — check-then-park, with the race available on demand.
//!
//! Every other spike here visualizes data. This one visualizes a *window in time*: the gap between
//! a reader's synchronous check and its park. On the real engine that gap has no duration, so the
//! only way to show why it matters is to open it deliberately and watch what breaks.

import { useState } from "react";
import {
  abortAll,
  beginCheck,
  cancel,
  createParkState,
  isQuiescent,
  isSettled,
  publish,
  runChecks,
  setThreading,
  type ParkState,
  type Reader,
  type ReaderState,
} from "./parkSim.js";

const STATE_COLOR: Record<ReaderState, string> = {
  checking: "var(--color-category-gray)",
  parked: "var(--color-category-blue)",
  satisfied: "var(--color-category-green)",
  orphaned_unreachable: "var(--color-category-orange)",
  orphaned_poisoned: "var(--color-category-purple)",
  lost_wakeup: "var(--color-category-red)",
};

const STATE_LABEL: Record<ReaderState, string> = {
  checking: "checking",
  parked: "parked",
  satisfied: "satisfied",
  orphaned_unreachable: "orphaned (unreachable)",
  orphaned_poisoned: "orphaned (poisoned)",
  lost_wakeup: "PARKED FOREVER",
};

const PRODUCERS = 3;
const TARGETS = [3, 2];

function fresh(threading: ParkState["threading"]): ParkState {
  return runChecks(createParkState(PRODUCERS, TARGETS, threading));
}

export function CheckThenParkSpike(): React.JSX.Element {
  const [state, setState] = useState<ParkState>(() => fresh("single"));

  const multi = state.threading === "multi";
  const windowOpen = state.midCheck.length > 0;
  const stuck = state.readers.some((r) => r.state === "lost_wakeup");

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-2xl font-extrabold">Check-then-park, and the race it ignores</h1>
      </div>
      <p className="mb-4 max-w-4xl text-sm text-ink-secondary">
        A reader waiting on a channel checks its arrival count, and if it is short, parks on a
        notify. The wake is <code>notify_waiters</code>, which wakes only readers{" "}
        <em>already parked</em> — a notify with nobody waiting is dropped, not queued. That is the
        classic lost-wakeup setup, and it is safe here for one reason: the engine runs
        current-thread, so nothing can execute between the check and the park. The window has no
        duration. Switch the model below to give it one.
      </p>

      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-1.5">
            <span className="mr-1 text-sm text-ink-tertiary">runtime</span>
            {(["single", "multi"] as const).map((mode) => (
              <button key={mode} type="button"
                onClick={() => setState((s) => setThreading(s, mode))}
                className={`rounded border px-2.5 py-1 text-xs font-semibold ${
                  state.threading === mode
                    ? "border-transparent bg-accent-primary text-black"
                    : "border-white/15 bg-surface-panel text-ink-secondary"}`}>
                {mode === "single" ? "single-threaded (real)" : "multi-threaded (hypothetical)"}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5">
            <button type="button"
              disabled={!multi || isQuiescent(state)}
              onClick={() => setState(beginCheck)}
              title={multi
                ? "Hold the readers between their check and their park"
                : "Impossible on a current-thread runtime — nothing can interleave here"}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary disabled:opacity-35">
              Open the check window
            </button>
            <button type="button" onClick={() => setState(() => fresh(state.threading))}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Reset
            </button>
            <button type="button" onClick={() => setState((s) => abortAll(s))}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              abort_all
            </button>
          </div>

          <div className="ml-auto flex items-center gap-5 text-sm tabular-nums">
            <span><span className="text-ink-tertiary">arrival</span> <strong>{state.arrival}</strong></span>
            <span><span className="text-ink-tertiary">remaining</span> <strong>{state.remaining}</strong></span>
            {state.poisoned && (
              <span className="rounded px-2 py-0.5 text-[11px] font-bold text-black"
                style={{ background: STATE_COLOR.orphaned_poisoned }}>POISONED</span>
            )}
            {windowOpen && (
              <span className="rounded px-2 py-0.5 text-[11px] font-bold text-black"
                style={{ background: "var(--color-category-yellow)" }}>CHECK WINDOW OPEN</span>
            )}
          </div>
        </div>
      </div>

      {stuck && (
        <div className="mb-4 rounded-lg border px-4 py-3 text-sm"
          style={{ borderColor: STATE_COLOR.lost_wakeup, background: "rgba(255,0,0,0.06)" }}>
          <strong style={{ color: STATE_COLOR.lost_wakeup }}>Deadlock.</strong>{" "}
          A wake fired while that reader was mid-check, so <code>notify_waiters</code> discarded it —
          there was nobody parked to receive it. The reader then parked into a silence no later
          write will break. Keep publishing: its target is met and it still never runs.
        </div>
      )}

      <div className="grid grid-cols-[1fr_1fr_360px] gap-4">
        <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
          <h2 className="mb-2 text-[10px] font-bold tracking-widest text-ink-secondary">
            PRODUCERS — declared for this channel
          </h2>
          <div className="flex flex-col gap-2">
            {state.producers.map((p) => (
              <div key={p.id} className="flex items-center gap-2">
                <span className="w-8 font-mono text-xs text-ink-tertiary">{p.id}</span>
                <button type="button" disabled={p.status !== "pending"}
                  onClick={() => setState((s) => publish(s, p.id))}
                  className="rounded border border-white/15 bg-surface-panel px-2 py-1 text-xs font-semibold disabled:opacity-30"
                  style={{ color: "var(--color-category-green)" }}>
                  publish
                </button>
                <button type="button" disabled={p.status !== "pending"}
                  onClick={() => setState((s) => cancel(s, p.id))}
                  className="rounded border border-white/15 bg-surface-panel px-2 py-1 text-xs font-semibold disabled:opacity-30"
                  style={{ color: "var(--color-category-orange)" }}>
                  cancel
                </button>
                <span className="text-xs text-ink-quaternary">{p.status}</span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-snug text-ink-quaternary">
            A publish raises <code>arrival</code> and lowers <code>remaining</code>; a cancel only
            lowers <code>remaining</code>. Both wake every parked reader.
          </p>
        </section>

        <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
          <h2 className="mb-2 text-[10px] font-bold tracking-widest text-ink-secondary">
            READERS — each re-checks its own count on wake
          </h2>
          <div className="flex flex-col gap-2">
            {state.readers.map((r) => (
              <ReaderCard key={r.id} reader={r} midCheck={state.midCheck.includes(r.id)} />
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-snug text-ink-quaternary">
            Two readers, one channel, different targets. Cancel a single producer and watch the
            three-target reader orphan <em>itself</em> while the two-target reader carries on — the
            channel is not poisoned, because a lower-count reader may still be satisfiable.
          </p>
        </section>

        <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
          <h2 className="mb-2 text-[10px] font-bold tracking-widest text-ink-secondary">
            THE LOOP — await_count, in order
          </h2>
          <pre className="overflow-x-auto font-mono text-[10px] leading-[1.6] text-ink-secondary">
{`loop {
  // single-threaded: nothing runs
  // between this check and the await
  let notify = {
    if arrival >= target {`}
            <b style={{ color: STATE_COLOR.satisfied }}>{` return Ok`}</b>{` }
    if orphaned {`}
            <b style={{ color: STATE_COLOR.orphaned_poisoned }}>{` return Err`}</b>{` }
    if arrival + remaining < target {
`}
            <b style={{ color: STATE_COLOR.orphaned_unreachable }}>{`      return Err(Orphaned)`}</b>{` }
    notifiers[channel].clone()
  };`}
            <b style={{ color: windowOpen ? "var(--color-category-yellow)" : "inherit" }}>
              {windowOpen ? "   ◄ you are here" : ""}
            </b>{`
  notify.`}
            <b style={{ color: STATE_COLOR.parked }}>{`notified().await`}</b>{`;
}`}
          </pre>
          <p className="mt-2 text-[11px] leading-snug text-ink-quaternary">
            Ported from <code>channel_store.rs:247</code>. The order matters: satisfied is checked
            before poisoned, so a reader whose count was already met is never orphaned by a late
            abort.
          </p>
        </section>
      </div>

      <section className="mt-4 rounded-lg border border-white/10 bg-surface-elevated p-3">
        <h2 className="mb-2 text-[10px] font-bold tracking-widest text-ink-secondary">EVENT LOG</h2>
        <div className="max-h-40 overflow-y-auto font-mono text-[10px] leading-[1.6]">
          {state.log.length === 0 && <span className="text-ink-quaternary">Nothing yet.</span>}
          {state.log.map((line, i) => (
            <div key={i} style={{ color: line.includes("WAKE LOST") ? STATE_COLOR.lost_wakeup : undefined }}>
              {line}
            </div>
          ))}
        </div>
      </section>

      <p className="mt-3 text-[11px] text-ink-quaternary">
        Modelled on <code>rust/runtime/src/graph/channel_store.rs</code>:{" "}
        <code>await_count</code> at :247, <code>notify_channel</code> at :301, the self-orphan
        comment at :268, and channel poisoning at :420. The multi-threaded model is hypothetical —
        the engine is current-thread with a <code>LocalSet</code>, which is exactly why the window
        it opens does not exist.
      </p>
    </div>
  );
}

function ReaderCard({ reader, midCheck }: { reader: Reader; midCheck: boolean }): React.JSX.Element {
  const color = STATE_COLOR[reader.state];
  return (
    <div className="rounded border p-2"
      style={{
        borderColor: midCheck ? "var(--color-category-yellow)" : "rgba(255,255,255,0.1)",
        background: midCheck ? "rgba(255,255,0,0.05)" : undefined,
      }}>
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xs font-bold text-ink-primary">
          {reader.id} <span className="text-ink-tertiary">target {reader.target}</span>
        </span>
        <span className="rounded px-1.5 py-0.5 text-[10px] font-bold text-black" style={{ background: color }}>
          {midCheck ? "MID-CHECK" : STATE_LABEL[reader.state]}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-ink-quaternary">
        {reader.note}
        {reader.rechecks > 0 && !isSettled(reader) && ` · ${reader.rechecks} re-checks`}
      </div>
    </div>
  );
}
