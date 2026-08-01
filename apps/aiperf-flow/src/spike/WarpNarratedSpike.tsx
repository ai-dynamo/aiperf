/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the warp, narrated. The voice drives the playhead, not a free-running timer.
//!
//! This is the slide-clock idea made concrete on one chart before building it as a module. Each
//! beat authors where the playhead should be by the time its narration finishes; position inside
//! a beat comes from the spoken word position, so the picture and the voice cannot drift apart
//! however fast the chosen voice speaks.

import { useEffect, useMemo, useRef, useState } from "react";
import { PresentationShell } from "../shell/PresentationShell.js";
import { useNarratedDeck } from "../audio/index.js";
import { splitWords } from "../audio/narration.js";
import type { SlideDefinition } from "../deck/types.js";
import { buildTrace, buildWarpMap, rawTimeFor } from "./warpTrace.js";
import { WarpTracks, warpSummary } from "./WarpTracks.js";
import { idleGaps } from "../decks/weka-timing-transforms-interactive/logic.js";

const SEED = 3;
const TRACE_MS = 75_000;
const CAP = 1.2;

/**
 * One narrated beat, plus where the warped playhead should have reached by its *final word*.
 *
 * A fraction of the warped span rather than a number of seconds, so re-seeding the trace or moving
 * the cap does not invalidate the script.
 */
type Beat = { endAt: number; narration: string; title: string; lede: string; caption: string };

const BEATS: Beat[] = [
  {
    endAt: 0.06,
    title: "A recorded agent session",
    lede: "Seventy seconds of real work, on top. Every bar is one request.",
    caption: "Trace frozen from the agent simulation; lanes are main and its subagents.",
    narration:
      "Here is a recorded agent session. Each bar is one request, and each row is one agent — the main agent on top, the subagents it spawned beneath it. Nothing has been changed yet. This is simply what happened, laid out on the clock it happened on.",
  },
  {
    endAt: 0.22,
    title: "Most of it is waiting",
    lede: "The orange bands are stretches where nothing at all was in flight.",
    caption: "idleGaps() classifies every true gap: running-max end to the next start.",
    narration:
      "Now look at how little of it is actually busy. The orange bands are dead air — stretches where no request was in flight anywhere in the session. A human was reading, or a tool was running, or the agent was simply thinking. Replaying this verbatim would mean sitting through every one of those gaps at full length, and most of the wall clock would be spent doing nothing.",
  },
  {
    endAt: 0.5,
    title: "Cap the dead air, never the request",
    lede: "Each gap is cut to the cap. The bars themselves are untouched.",
    caption: "The cut table shifts every later timestamp left by the cumulative excess.",
    narration:
      "So the gaps get capped. Only the dead air, never the requests. Watch the bottom track: it is the same session with each idle stretch trimmed down to just over a second. The requests are all still there, in the same order, and — this is the part that matters — every bar is exactly as wide as it was. Service time is never compressed, because compressing it would change the thing we are trying to measure.",
  },
  {
    endAt: 0.82,
    title: "The two clocks separate",
    lede: "Same session, two playheads, drifting further apart at every gap.",
    caption: "The raw head accelerates through each collapsed gap; the warped head is steady.",
    narration:
      "And here is the effect. Both playheads are moving through the same session, but the top one has to cross all that dead air while the bottom one skips it. Every time they hit a capped gap, the raw head races ahead and the gap between them widens. By this point in the replay the recorded clock is more than twenty seconds ahead of the clock the runtime is actually issuing on.",
  },
  {
    endAt: 1,
    title: "Half the session, never replayed",
    lede: "Same work, same order, same durations — in half the wall time.",
    caption: "Compression is a property of the trace: sparser sessions compress harder.",
    narration:
      "By the end, the warped track finishes in about half the wall time of the recording. The green block on the right is the part that is never replayed at all. Same requests, same order, same durations — the only thing removed is the waiting. That is the whole transform, and it is why a benchmark can replay a long agent session without spending an afternoon on it.",
  },
];

/** `PresentationShell` reads narration, title, lede, and caption; the diagram is our own child. */
const SLIDES: readonly SlideDefinition[] = BEATS.map((b, i) => ({
  id: `beat-${i}`,
  eyebrow: `${String(i + 1).padStart(2, "0")} · THE WARP`,
  title: b.title,
  lede: b.lede,
  narration: b.narration,
  caption: b.caption,
  nodes: [],
  edges: [],
}));

export function WarpNarratedSpike(): React.JSX.Element {
  const trace = useMemo(() => buildTrace(SEED, TRACE_MS), []);
  const { warpSpan } = useMemo(() => warpSummary(trace, CAP), [trace]);
  const warpMap = useMemo(() => buildWarpMap(idleGaps(trace.reqs, CAP), CAP), [trace]);

  const narrated = useNarratedDeck({
    narrations: BEATS.map((b) => b.narration),
    storagePrefix: "spike:warp-narrated",
  });

  // Virtual time only ever moves forward. `activeWordIndex` does not: `speakNarration` drives
  // word events from estimated timers and onboundary at once, so a voice slower than the estimate
  // reports a lower index than an already-fired timer. Left unclamped, the playhead would rewind.
  const highWater = useRef(0);
  const shownRef = useRef(0);
  const [, force] = useState(0);

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const dt = Math.min(64, t - last);
      last = t;
      const k = 1 - Math.exp(-dt / 160);
      const next = shownRef.current + (highWater.current - shownRef.current) * k;
      if (Math.abs(next - shownRef.current) > 1e-4) {
        shownRef.current = next;
        force((n) => n + 1);
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const beat = BEATS[narrated.index] ?? BEATS[0]!;
  const from = (BEATS[narrated.index - 1]?.endAt ?? 0) * warpSpan;
  const to = beat.endAt * warpSpan;
  const words = splitWords(beat.narration).length;
  const within = narrated.activeWordIndex < 0 ? 0 : Math.min(1, narrated.activeWordIndex / Math.max(1, words - 1));
  const target = from + (to - from) * within;

  if (target > highWater.current) {
    highWater.current = target;
  } else if (narrated.index === 0 && narrated.activeWordIndex <= 0) {
    // A genuine restart, as opposed to a backwards word correction.
    highWater.current = 0;
  }
  // Narration sets the target; the frame loop above is what the viewer actually sees.
  const warpedNow = shownRef.current;
  const rawNow = rawTimeFor(warpedNow, warpMap);

  return (
    <PresentationShell
      slides={SLIDES}
      slideIndex={narrated.index}
      onSlideIndexChange={narrated.goTo}
      narrated={narrated}
      title="The warp, narrated"
    >
      <div className="flex h-full flex-col px-6 pt-4">
        <div className="mb-2 flex items-baseline gap-5 text-sm tabular-nums">
          <span><span className="text-ink-tertiary">raw</span>{" "}
            <strong>{rawNow.toFixed(1)}s</strong>
            <span className="text-ink-quaternary"> / {trace.rawSpan.toFixed(1)}</span></span>
          <span><span className="text-ink-tertiary">warped</span>{" "}
            <strong style={{ color: "var(--color-category-green)" }}>{warpedNow.toFixed(1)}s</strong>
            <span className="text-ink-quaternary"> / {warpSpan.toFixed(1)}</span></span>
          <span><span className="text-ink-tertiary">ahead by</span>{" "}
            <strong style={{ color: "var(--color-category-orange)" }}>
              {(rawNow - warpedNow).toFixed(1)}s
            </strong></span>
        </div>
        <WarpTracks trace={trace} cap={CAP} rawNow={rawNow} warpedNow={warpedNow} revealWithHead />
      </div>
    </PresentationShell>
  );
}
