// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import {
  activeCausalBeat,
  adjacentCausalBeat,
  causalBeatState,
  type CausalBeat,
} from "../causal-replay.js";

/** Labelled causal-beat navigation control for exact virtual-time seeks. */
export type CausalPathProps = Readonly<{
  beats: readonly CausalBeat[];
  timeMs: number;
  onSeek(timeMs: number, beatId: string): void;
}>;

function beatById(
  beats: readonly CausalBeat[],
  id: string | null,
): CausalBeat | null {
  if (id === null) {
    return null;
  }
  return beats.find((beat) => beat.id === id) ?? null;
}

function defaultFocusedId(
  beats: readonly CausalBeat[],
  timeMs: number,
): string | null {
  return activeCausalBeat(beats, timeMs)?.id ?? beats[0]?.id ?? null;
}

/**
 * Causal Replay path: one keyboard-operable navigation control over authored
 * beats. Seeks the integer virtual clock; never a free-form range scrubber.
 */
export function CausalPath({
  beats,
  timeMs,
  onSeek,
}: CausalPathProps): ReactNode {
  const labelId = useId();
  const descriptionPrefix = useId();
  const rootRef = useRef<HTMLElement | null>(null);
  const buttonRefs = useRef(new Map<string, HTMLButtonElement>());
  const [focusedId, setFocusedId] = useState<string | null>(() =>
    defaultFocusedId(beats, timeMs),
  );
  const focusMovePending = useRef(false);

  const active = activeCausalBeat(beats, timeMs);
  const resolvedFocusId =
    beatById(beats, focusedId)?.id ?? defaultFocusedId(beats, timeMs);

  useEffect(() => {
    if (resolvedFocusId === focusedId) {
      return;
    }
    setFocusedId(resolvedFocusId);
  }, [focusedId, resolvedFocusId]);

  useEffect(() => {
    const activeId = active?.id;
    if (activeId === undefined) {
      return;
    }
    const root = rootRef.current;
    const focusInside =
      root !== null &&
      document.activeElement instanceof Node &&
      root.contains(document.activeElement);
    if (!focusInside) {
      setFocusedId(activeId);
    }
  }, [active?.id]);

  useEffect(() => {
    if (!focusMovePending.current || resolvedFocusId === null) {
      return;
    }
    focusMovePending.current = false;
    buttonRefs.current.get(resolvedFocusId)?.focus();
  }, [resolvedFocusId]);

  function seekTo(beat: CausalBeat): void {
    setFocusedId(beat.id);
    onSeek(beat.timeMs, beat.id);
  }

  function moveFocus(
    direction: "first" | "previous" | "next" | "last",
  ): void {
    const next = adjacentCausalBeat(beats, resolvedFocusId, direction);
    if (next === null) {
      return;
    }
    focusMovePending.current = true;
    seekTo(next);
  }

  function onBeatKeyDown(event: KeyboardEvent<HTMLButtonElement>): void {
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowUp":
        event.preventDefault();
        moveFocus("previous");
        break;
      case "ArrowRight":
      case "ArrowDown":
        event.preventDefault();
        moveFocus("next");
        break;
      case "Home":
        event.preventDefault();
        moveFocus("first");
        break;
      case "End":
        event.preventDefault();
        moveFocus("last");
        break;
      default:
        break;
    }
  }

  return (
    <nav
      aria-labelledby={labelId}
      className="aiperf-flow__causal-path"
      data-beat-count={beats.length}
      data-current-beat={active?.id ?? undefined}
      ref={rootRef}
    >
      <h2 className="aiperf-flow__causal-path-label" id={labelId}>
        Causal path
      </h2>
      <ol aria-label="Causal beats" className="aiperf-flow__causal-beats">
        {beats.map((beat, index) => {
          const state = causalBeatState(beat, timeMs);
          const isCurrent = active?.id === beat.id;
          const isFocused = resolvedFocusId === beat.id;
          const descriptionId =
            beat.description === undefined
              ? undefined
              : `${descriptionPrefix}-${beat.id}`;

          return (
            <li key={beat.id}>
              <button
                aria-current={isCurrent ? "step" : undefined}
                aria-describedby={descriptionId}
                aria-label={beat.label}
                aria-posinset={index + 1}
                aria-setsize={beats.length}
                className="aiperf-flow__causal-beat"
                data-beat-id={beat.id}
                data-beat-source={beat.source}
                data-state={state}
                data-time-ms={beat.timeMs}
                onClick={() => {
                  focusMovePending.current = true;
                  seekTo(beat);
                }}
                onFocus={() => {
                  setFocusedId(beat.id);
                }}
                onKeyDown={onBeatKeyDown}
                ref={(node) => {
                  if (node === null) {
                    buttonRefs.current.delete(beat.id);
                    return;
                  }
                  buttonRefs.current.set(beat.id, node);
                }}
                tabIndex={isFocused ? 0 : -1}
                type="button"
              >
                {beat.label}
              </button>
              {beat.description === undefined ? null : (
                <p
                  className="aiperf-flow__causal-beat-description"
                  id={descriptionId}
                >
                  {beat.description}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
