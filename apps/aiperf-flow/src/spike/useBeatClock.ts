/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — narration progress as a position along an authored span.
//!
//! Extracted once a second narrated spike needed it. Everything awkward about driving a picture
//! from a voice lives here: the stale word index at a beat boundary, the fact that the word index
//! is not monotonic, and the fact that word events are far too coarse to animate from directly.
//!
//! This is the shape `src/clock/` would take, proven on two consumers first.

import { useEffect, useRef, useState } from "react";
import { splitWords } from "../audio/narration.js";

/** One narrated beat and where the position should be by its *final* word, as a 0..1 fraction. */
export type BeatAnchor = { endAt: number; narration: string };

/**
 * Position implied by a beat and a word index, in fractions of the span.
 *
 * Pure so the boundary arithmetic can be tested without a voice. `valid` is false on the render
 * where the beat has advanced but the word index still belongs to the previous narration.
 */
export function beatFraction(
  beats: readonly BeatAnchor[],
  index: number,
  activeWordIndex: number,
  valid: boolean,
): number {
  const beat = beats[index];
  if (beat === undefined) return 0;
  const from = beats[index - 1]?.endAt ?? 0;
  const words = splitWords(beat.narration).length;
  const within =
    !valid || activeWordIndex < 0 ? 0 : Math.min(1, activeWordIndex / Math.max(1, words - 1));
  return from + (beat.endAt - from) * within;
}

export type BeatClock = {
  /** Eased position in span units — what to draw. */
  position: number;
  /** Un-eased narration-implied position, for anything that must not lag. */
  target: number;
};

/**
 * Drive a position along `span` from a narrated deck's word progress.
 *
 * @param beats - Authored anchors, ascending by `endAt`.
 * @param index - Current beat, from `useNarratedDeck`.
 * @param activeWordIndex - Live word position; `-1` while idle.
 * @param span - Total the fractions are of, in whatever unit the caller draws in.
 * @param tauMs - Easing time constant. Word events land every few hundred ms, which steps
 *   visibly without this.
 */
export function useBeatClock(
  beats: readonly BeatAnchor[],
  index: number,
  activeWordIndex: number,
  span: number,
  tauMs = 160,
): BeatClock {
  const highWater = useRef(0);
  const shown = useRef(0);
  const lastBeat = useRef(0);
  /** False from a beat change until the word index has demonstrably reset for the new narration. */
  const wordIndexValid = useRef(true);
  const [, force] = useState(0);

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const dt = Math.min(64, t - last);
      last = t;
      const k = 1 - Math.exp(-dt / tauMs);
      const next = shown.current + (highWater.current - shown.current) * k;
      if (Math.abs(next - shown.current) > 1e-4) {
        shown.current = next;
        force((n) => n + 1);
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, [tauMs]);

  // On the render where the beat advances, `activeWordIndex` still belongs to the previous
  // narration while the range is already the new beat's. Reading them together lands the target
  // most of the way through the new beat, which the monotonic clamp then latches — the position
  // lurches forward and then freezes until the real target climbs back to it.
  if (lastBeat.current !== index) {
    lastBeat.current = index;
    wordIndexValid.current = false;
    highWater.current = (index === 0 ? 0 : (beats[index - 1]?.endAt ?? 0)) * span;
  }
  if (!wordIndexValid.current && activeWordIndex <= 1) wordIndexValid.current = true;

  const target = beatFraction(beats, index, activeWordIndex, wordIndexValid.current) * span;
  // Virtual time only moves forward: `activeWordIndex` itself does not, because `speakNarration`
  // drives word events from estimated timers and `onboundary` at once.
  if (target > highWater.current) highWater.current = target;

  return { position: shown.current, target: highWater.current };
}
