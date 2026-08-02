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
  ControlBar,
  Legend,
  LegendItem,
  Meter,
  Note,
  Panel,
  Readout,
  SourceNote,
  SpikeHeader,
  Toggle,
} from "../ui.js";
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
      <SpikeHeader title="Check-then-park, and the race it ignores">
        <p>
          A reader waiting on a channel checks its arrival count, and if it is short, parks on a
          notify. The wake is <code>notify_waiters</code>, which wakes only readers{" "}
          <em>already parked</em> — a notify with nobody waiting is dropped, not queued.
        </p>
        <p>
          That is the classic lost-wakeup setup, and it is safe here for exactly one reason: the
          engine runs current-thread, so nothing can execute between the check and the park. The
          window has no duration. Switch the runtime below to give it one, then publish into it.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">runtime</span>
          {(["single", "multi"] as const).map((mode) => (
            <Toggle key={mode} active={state.threading === mode}
              onClick={() => setState((s) => setThreading(s, mode))}>
              {mode === "single" ? "single-threaded (real)" : "multi-threaded (hypothetical)"}
            </Toggle>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <Toggle disabled={!multi || isQuiescent(state)} onClick={() => setState(beginCheck)}
            title={multi
              ? "Hold the readers between their check and their park"
              : "Impossible on a current-thread runtime — nothing can interleave here"}>
            Open the check window
          </Toggle>
          <Toggle onClick={() => setState(() => fresh(state.threading))}>Reset</Toggle>
          <Toggle onClick={() => setState((s) => abortAll(s))}>abort_all</Toggle>
        </div>

        <div className="ml-auto flex items-center gap-5">
          <Readout label="arrival" value={state.arrival} />
          <Readout label="remaining" value={state.remaining} />
          {state.poisoned && (
            <span className="rounded px-2.5 py-0.5 text-[13px] font-bold text-black"
              style={{ background: STATE_COLOR.orphaned_poisoned }}>POISONED</span>
          )}
          {windowOpen && (
            <span className="rounded px-2.5 py-0.5 text-[13px] font-bold text-black"
              style={{ background: "var(--color-category-yellow)" }}>CHECK WINDOW OPEN</span>
          )}
        </div>
      </ControlBar>

      {stuck && (
        <div className="mb-4 rounded-lg border px-5 py-4 text-base leading-relaxed"
          style={{ borderColor: STATE_COLOR.lost_wakeup, background: "rgba(255,0,0,0.06)" }}>
          <strong style={{ color: STATE_COLOR.lost_wakeup }}>Deadlock.</strong>{" "}
          A wake fired while that reader was mid-check, so <code>notify_waiters</code> discarded it —
          there was nobody parked to receive it. The reader then parked into a silence no later
          write will break. Keep publishing: its target is met and it still never runs.
        </div>
      )}

      <div className="grid grid-cols-[1fr_1.15fr_400px] gap-4">
        <Panel label="PRODUCERS" hint="declared for this channel">
          <div className="flex flex-col gap-2">
            {state.producers.map((p) => (
              <div key={p.id} className="flex items-center gap-2.5">
                <span className="w-8 font-mono text-[15px] text-ink-tertiary">{p.id}</span>
                <button type="button" disabled={p.status !== "pending"}
                  onClick={() => setState((s) => publish(s, p.id))}
                  className="rounded border border-white/15 bg-surface-panel px-2.5 py-1 text-[14px] font-semibold disabled:opacity-30"
                  style={{ color: "var(--color-category-green)" }}>
                  publish
                </button>
                <button type="button" disabled={p.status !== "pending"}
                  onClick={() => setState((s) => cancel(s, p.id))}
                  className="rounded border border-white/15 bg-surface-panel px-2.5 py-1 text-[14px] font-semibold disabled:opacity-30"
                  style={{ color: "var(--color-category-orange)" }}>
                  cancel
                </button>
                <span className="text-[14px] text-ink-quaternary">{p.status}</span>
              </div>
            ))}
          </div>
          <Note>
            A publish raises <code>arrival</code> and lowers <code>remaining</code>; a cancel only
            lowers <code>remaining</code>. Both wake every parked reader.
          </Note>
        </Panel>

        <Panel label="READERS" hint="each re-checks its own count on wake">
          <Legend>
            <LegendItem mark="▰" color={STATE_COLOR.satisfied}>arrived</LegendItem>
            <LegendItem mark="▱">still needed</LegendItem>
          </Legend>
          <div className="flex flex-col gap-2">
            {state.readers.map((r) => (
              <ReaderCard key={r.id} reader={r} arrival={state.arrival}
                midCheck={state.midCheck.includes(r.id)} />
            ))}
          </div>
          <Note>
            Two readers, one channel, different targets. Cancel a single producer and watch the
            three-target reader orphan <em>itself</em> while the two-target reader carries on — the
            channel is not poisoned, because a lower-count reader may still be satisfiable.
          </Note>
        </Panel>

        <Panel label="THE LOOP" hint="await_count, in order">
          <pre className="overflow-x-auto font-mono text-[13px] leading-[1.65] text-ink-secondary">
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
              {windowOpen ? "   \u25c4 you are here" : ""}
            </b>{`
  notify.`}
            <b style={{ color: STATE_COLOR.parked }}>{`notified().await`}</b>{`;
}`}
          </pre>
          <Note>
            Ported from <code>channel_store.rs:247</code>. The order matters: satisfied is checked
            before poisoned, so a reader whose count was already met is never orphaned by a late
            abort.
          </Note>
        </Panel>
      </div>

      <Panel label="EVENT LOG" className="mt-4">
        <div className="max-h-44 overflow-y-auto font-mono text-[14px] leading-[1.7]">
          {state.log.length === 0 && <span className="text-ink-quaternary">Nothing yet.</span>}
          {state.log.map((line, i) => (
            <div key={i} style={{ color: line.includes("WAKE LOST") ? STATE_COLOR.lost_wakeup : undefined }}>
              {line}
            </div>
          ))}
        </div>
      </Panel>

      <SourceNote>
        Modelled on <code>rust/runtime/src/graph/channel_store.rs</code>:{" "}
        <code>await_count</code> at :247, <code>notify_channel</code> at :301, the self-orphan
        comment at :268, and channel poisoning at :420. The multi-threaded model is hypothetical —
        the engine is current-thread with a <code>LocalSet</code>, which is exactly why the window
        it opens does not exist.
      </SourceNote>
    </div>
  );
}

function ReaderCard({
  reader,
  midCheck,
  arrival,
}: {
  reader: Reader;
  midCheck: boolean;
  arrival: number;
}): React.JSX.Element {
  const color = STATE_COLOR[reader.state];
  return (
    <div className="rounded border p-2.5"
      style={{
        borderColor: midCheck ? "var(--color-category-yellow)" : "rgba(255,255,255,0.1)",
        background: midCheck ? "rgba(255,255,0,0.05)" : undefined,
      }}>
      <div className="flex items-center justify-between gap-3">
        <span className="font-mono text-[15px] font-bold text-ink-primary">{reader.id}</span>
        {/* The gap to the target, drawn: filled cells have arrived, dashed ones have not. */}
        <Meter value={Math.min(arrival, reader.target)} target={reader.target} color={color} />
        <span className="font-mono text-[14px] tabular-nums text-ink-tertiary">
          {Math.min(arrival, reader.target)}/{reader.target}
        </span>
        <span className="ml-auto rounded px-2 py-0.5 text-[12px] font-bold text-black"
          style={{ background: color }}>
          {midCheck ? "MID-CHECK" : STATE_LABEL[reader.state]}
        </span>
      </div>
      <div className="mt-1.5 text-[13px] text-ink-quaternary">
        {reader.note}
        {reader.rechecks > 0 && !isSettled(reader) && ` · ${reader.rechecks} re-checks`}
      </div>
    </div>
  );
}
