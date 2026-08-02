/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — check-then-park, and the race it is allowed to ignore.
//!
//! Modelled on `await_count` in `rust/runtime/src/graph/channel_store.rs:247`:
//!
//! ```ignore
//! loop {
//!     // Check-then-park. Single-threaded: nothing runs between this
//!     // synchronous check and the `.notified().await` below, so a
//!     // `notify_waiters` from a concurrent writer can't be lost.
//!     let notify = {
//!         let inner = self.inner.borrow();
//!         if arrival >= target { return Ok(()); }
//!         if orphaned { return Err(Orphaned); }
//!         if arrival + remaining < target { return Err(Orphaned{..}); }
//!         inner.notifiers[channel].clone()
//!     };
//!     notify.notified().await;
//! }
//! ```
//!
//! Two things make it correct, and neither is a lock.
//!
//! The wake is `notify_waiters`, which wakes only readers *already parked* — a notify with nobody
//! waiting is dropped, not stored. On its own that is the classic lost-wakeup setup.
//!
//! What saves it is the execution model. The engine runs current-thread with a `LocalSet`, so no
//! other task can run between the synchronous check and the `.await`. The window a multi-threaded
//! executor would have simply does not exist here. This model can open it on demand, which is the
//! point of the spike: the guarantee is a property of the runtime, not of the data structure.

/** Where a reader is in the loop. */
export type ReaderState =
  | "checking"
  | "parked"
  | "satisfied"
  /** `arrival + remaining < target` — this reader alone can never be met. */
  | "orphaned_unreachable"
  /** The channel itself was poisoned: no producer will ever write. */
  | "orphaned_poisoned"
  /** A wake landed while this reader was mid-check, and was dropped. Parked forever. */
  | "lost_wakeup";

export type Reader = {
  id: string;
  /** How many arrivals this reader needs — the `k` of a k-of-n join. */
  target: number;
  state: ReaderState;
  /** Times this reader has woken and re-checked. */
  rechecks: number;
  note: string;
};

export type Producer = {
  id: string;
  /** Published, cancelled, or still outstanding. */
  status: "pending" | "published" | "cancelled";
};

/** Single-threaded is the real engine. Multi-threaded opens the window, hypothetically. */
export type ThreadingModel = "single" | "multi";

export type ParkState = {
  /** Arrivals committed to the channel. */
  arrival: number;
  /** Producers that have neither published nor cancelled. */
  remaining: number;
  producers: Producer[];
  readers: Reader[];
  /** Set when every producer is gone with nothing written and no init seed. */
  poisoned: boolean;
  hasInitSeed: boolean;
  threading: ThreadingModel;
  /** Readers currently mid-check, i.e. exposed to a lost wake under `multi`. */
  midCheck: string[];
  log: string[];
};

export function createParkState(
  producerCount = 3,
  targets: readonly number[] = [3, 2],
  threading: ThreadingModel = "single",
): ParkState {
  return {
    arrival: 0,
    remaining: producerCount,
    producers: Array.from({ length: producerCount }, (_, i) => ({
      id: `p${i}`,
      status: "pending",
    })),
    readers: targets.map((target, i) => ({
      id: `r${i}`,
      target,
      state: "checking",
      rechecks: 0,
      note: "",
    })),
    poisoned: false,
    hasInitSeed: false,
    threading,
    midCheck: [],
    log: [],
  };
}

/** Has this reader reached a terminal state? */
export function isSettled(reader: Reader): boolean {
  return (
    reader.state === "satisfied" ||
    reader.state === "orphaned_unreachable" ||
    reader.state === "orphaned_poisoned" ||
    reader.state === "lost_wakeup"
  );
}

/**
 * The synchronous half of the loop: the three early exits, in the order the Rust performs them.
 *
 * Returns the state the reader lands in. `parked` means none of the exits fired and it is about to
 * `.await` — the instant that matters.
 */
export function evaluate(state: ParkState, reader: Reader): ReaderState {
  if (state.arrival >= reader.target) return "satisfied";
  if (state.poisoned) return "orphaned_poisoned";
  // This reader's count can no longer be met. Note it orphans *itself* — the channel stays live,
  // because a lower-count reader on the same channel may still be satisfiable.
  if (state.arrival + state.remaining < reader.target) return "orphaned_unreachable";
  return "parked";
}

function noteFor(state: ReaderState, reader: Reader, s: ParkState): string {
  switch (state) {
    case "satisfied":
      return `arrival ${s.arrival} ≥ target ${reader.target}`;
    case "orphaned_poisoned":
      return "channel poisoned: all producers cancelled";
    case "orphaned_unreachable":
      return `arrival ${s.arrival} + remaining ${s.remaining} < target ${reader.target}`;
    case "lost_wakeup":
      return "woken while mid-check; notify_waiters dropped it";
    case "parked":
      return `waiting for ${reader.target - s.arrival} more`;
    default:
      return "";
  }
}

/** Advance every unsettled reader through one synchronous check. */
export function runChecks(input: ParkState): ParkState {
  const log = [...input.log];
  const readers = input.readers.map((reader) => {
    if (isSettled(reader)) return reader;
    const next = evaluate(input, reader);
    if (next !== reader.state) {
      log.push(`${reader.id}: ${reader.state} → ${next}`);
    }
    return { ...reader, state: next, note: noteFor(next, reader, input) };
  });
  return { ...input, readers, midCheck: [], log };
}

/**
 * Put readers into the check window without completing it.
 *
 * Only reachable under `multi`: on a current-thread runtime nothing can interleave here, so the
 * window has no duration and this is not a state the real engine can be in.
 */
export function beginCheck(input: ParkState): ParkState {
  if (input.threading === "single") return input;
  const midCheck = input.readers.filter((r) => !isSettled(r)).map((r) => r.id);
  return {
    ...input,
    midCheck,
    log: [...input.log, `${midCheck.join(", ")}: entered the check window`],
  };
}

/**
 * A producer publishes: arrival climbs, remaining falls, and parked readers are woken.
 *
 * `notify_waiters` wakes only readers *already parked*. A reader still mid-check is not waiting
 * yet, so its wake is dropped — and when it does park, nothing will ever wake it again.
 */
export function publish(input: ParkState, producerId: string): ParkState {
  const producer = input.producers.find((p) => p.id === producerId);
  if (producer === undefined || producer.status !== "pending") return input;

  const producers = input.producers.map((p) =>
    p.id === producerId ? { ...p, status: "published" as const } : p,
  );
  const arrival = input.arrival + 1;
  const remaining = input.remaining - 1;
  const log = [...input.log, `${producerId} published → arrival ${arrival}, remaining ${remaining}`];

  const readers = input.readers.map((reader) => {
    if (isSettled(reader)) return reader;
    if (input.midCheck.includes(reader.id)) {
      // The wake arrives while this reader is between its check and its park. `notify_waiters`
      // has no queue, so it is discarded and the reader parks into a silence.
      log.push(`${reader.id}: WAKE LOST — mid-check when notify_waiters fired`);
      return { ...reader, state: "lost_wakeup" as const, note: noteFor("lost_wakeup", reader, input) };
    }
    if (reader.state !== "parked") return reader;
    return { ...reader, rechecks: reader.rechecks + 1 };
  });

  const woken = readers.filter((r) => r.state === "parked").map((r) => r.id);
  if (woken.length > 0) log.push(`notify_waiters woke ${woken.join(", ")} — each re-checks`);

  return runChecks({ ...input, producers, arrival, remaining, readers, midCheck: [], log });
}

/**
 * A producer is cancelled: it will never write.
 *
 * The channel is poisoned only when nothing can ever arrive — no producer left, nothing written,
 * no init seed. Otherwise individual readers orphan themselves and the channel stays usable.
 */
export function cancel(input: ParkState, producerId: string): ParkState {
  const producer = input.producers.find((p) => p.id === producerId);
  if (producer === undefined || producer.status !== "pending") return input;

  const producers = input.producers.map((p) =>
    p.id === producerId ? { ...p, status: "cancelled" as const } : p,
  );
  const remaining = input.remaining - 1;
  const poisoned =
    input.poisoned || (remaining === 0 && input.arrival === 0 && !input.hasInitSeed);
  const log = [...input.log, `${producerId} cancelled → remaining ${remaining}`];
  if (poisoned && !input.poisoned) log.push("channel poisoned: all_producers_cancelled");

  const readers = input.readers.map((reader) => {
    if (isSettled(reader)) return reader;
    if (input.midCheck.includes(reader.id)) {
      log.push(`${reader.id}: WAKE LOST — mid-check when notify_waiters fired`);
      return { ...reader, state: "lost_wakeup" as const, note: noteFor("lost_wakeup", reader, input) };
    }
    if (reader.state !== "parked") return reader;
    return { ...reader, rechecks: reader.rechecks + 1 };
  });

  return runChecks({ ...input, producers, remaining, poisoned, readers, midCheck: [], log });
}

/** Fail-fast: poison the channel and wake everyone to observe it. */
export function abortAll(input: ParkState, reason = "fail_fast"): ParkState {
  const log = [...input.log, `abort_all(${reason}) — every channel poisoned`];
  return runChecks({ ...input, poisoned: true, midCheck: [], log });
}

export function setThreading(input: ParkState, threading: ThreadingModel): ParkState {
  return {
    ...input,
    threading,
    midCheck: threading === "single" ? [] : input.midCheck,
    log: [...input.log, `threading model → ${threading}`],
  };
}

/** True once no reader can make further progress. */
export function isQuiescent(state: ParkState): boolean {
  return state.readers.every(isSettled);
}
