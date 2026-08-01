/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";
import { inkClassName, strokeClassName, surfaceClassName } from "../theme/tokens.js";
import { TopBar } from "./TopBar.js";
import type { SlideDefinition } from "../deck/types.js";
import type { NarratedDeck } from "../audio/index.js";
import { formatDeckDuration, formatStepDuration } from "../audio/index.js";
import { PlaybackControls } from "./PlaybackControls.js";
import { StartGate } from "./StartGate.js";
import { Subtitles } from "./Subtitles.js";

export function PresentationShell({
  slides,
  slideIndex,
  onSlideIndexChange,
  narrated,
  title,
  children,
}: {
  slides: readonly SlideDefinition[];
  slideIndex: number;
  onSlideIndexChange: (next: number) => void;
  /**
   * Autoplay narration from {@link useNarratedDeck}. When omitted the shell stays
   * a manual click-through deck, exactly as before.
   */
  narrated?: NarratedDeck;
  /** Deck title for the start gate. Only used when `narrated` is provided. */
  title?: string;
  children: ReactNode;
}): React.JSX.Element {
  const [showNotes, setShowNotes] = useState(false);
  const shellRef = useRef<HTMLDivElement>(null);
  const started = narrated?.started ?? false;

  // Dismissing the start gate unmounts the button that had focus, which drops
  // focus back to <body> — where the shell's keydown handler never sees it. Without
  // this, arrow and space keys silently do nothing until the viewer clicks the deck.
  useEffect(() => {
    if (started) shellRef.current?.focus();
  }, [started]);

  const slide = slides[slideIndex];
  if (slide === undefined) {
    throw new Error(`slideIndex ${slideIndex} out of range for ${slides.length} slides.`);
  }

  const narrations = slides.map((s) => s.narration);
  const isLast = slideIndex === slides.length - 1;

  return (
    // The keyboard handler needs a focusable host; `relative` anchors the start gate.
    <div
      ref={shellRef}
      className={clsx(
        "relative flex h-screen flex-col focus:outline-none",
        surfaceClassName("chrome"),
      )}
      tabIndex={narrated ? 0 : undefined}
      onKeyDown={narrated?.onKeyDown}
    >
      {/* The only route back to the deck index — without it a deck is a dead end. */}
      <TopBar section={title ?? "Deck"} />
      {narrated && !narrated.started && (
        <StartGate
          title={title ?? "This walkthrough"}
          speechAvailable={narrated.speechAvailable}
          voices={narrated.voices}
          selectedVoiceURI={narrated.voiceURI}
          onVoiceSelect={narrated.setVoiceURI}
          onStartWithNarration={() => narrated.begin(true)}
          onStartSilent={() => narrated.begin(false)}
        />
      )}

      <div className={clsx("flex gap-1.5 border-b px-4 py-3", strokeClassName("secondary"))}>
        {slides.map((s, index) => (
          <button
            key={s.id}
            type="button"
            aria-label={`Go to slide ${index + 1}: ${s.title}`}
            // Colour alone carried "which slide am I on"; aria-current says it too.
            aria-current={index === slideIndex ? "step" : undefined}
            onClick={() => onSlideIndexChange(index)}
            className={clsx(
              "h-[3px] flex-1 rounded-full transition-colors",
              index === slideIndex
                ? "bg-accent-primary"
                : "bg-[--color-stroke-secondary] hover:bg-[--color-stroke-primary]",
            )}
          />
        ))}
      </div>

      <div className="min-h-0 flex-1">{children}</div>

      {narrated ? (
        <Subtitles
          text={slide.narration}
          activeWordIndex={narrated.activeWordIndex}
          visible={narrated.started}
        />
      ) : (
        <div className={clsx("border-t px-4 py-2", strokeClassName("secondary"))}>
          <div className={clsx("text-sm", inkClassName("secondary"))}>{slide.narration}</div>
        </div>
      )}

      {showNotes && (
        <div
          id="deck-speaker-notes"
          className={clsx("border-t px-4 py-2", strokeClassName("secondary"))}
        >
          <div className={clsx("text-xs font-semibold", inkClassName("tertiary"))}>
            Speaker notes
          </div>
          <div className={clsx("text-sm", inkClassName("secondary"))}>{slide.caption}</div>
          {narrated && (
            <div className={clsx("mt-1 text-xs", inkClassName("tertiary"))}>
              Slide ~{formatStepDuration(slide.narration, narrated.speed)} · total{" "}
              {formatDeckDuration(narrations, narrated.speed)}
            </div>
          )}
        </div>
      )}

      <div
        className={clsx(
          "flex items-center justify-between border-t px-4 py-2",
          strokeClassName("secondary"),
        )}
      >
        <button
          type="button"
          disabled={slideIndex === 0}
          onClick={() => onSlideIndexChange(slideIndex - 1)}
          className={clsx(
            "px-3 py-1 text-sm font-semibold tracking-wide transition-colors hover:text-accent-primary disabled:opacity-40 disabled:hover:text-ink-primary",
            inkClassName("primary"),
          )}
        >
          ← Back
        </button>

        {narrated && (
          <PlaybackControls
            playing={narrated.playing}
            isLast={isLast}
            narrationEnabled={narrated.narrationEnabled}
            speechAvailable={narrated.speechAvailable}
            speed={narrated.speed}
            onTogglePlayback={narrated.togglePlayback}
            onToggleNarration={() => narrated.setNarrationEnabled(!narrated.narrationEnabled)}
            onSpeedChange={narrated.setSpeed}
          />
        )}

        <span className={clsx("text-xs font-medium tracking-wide", inkClassName("tertiary"))}>
          {slideIndex + 1} / {slides.length}
        </span>
        <div className="flex gap-2">
          <button
            type="button"
            aria-expanded={showNotes}
            aria-controls="deck-speaker-notes"
            onClick={() => setShowNotes((value) => !value)}
            className={clsx(
              "px-3 py-1 text-xs font-medium uppercase tracking-wide transition-colors hover:text-ink-primary",
              inkClassName("secondary"),
            )}
          >
            Speaker notes
          </button>
          <button
            type="button"
            disabled={isLast}
            onClick={() => onSlideIndexChange(slideIndex + 1)}
            className={clsx(
              "px-3 py-1 text-sm font-semibold tracking-wide transition-colors hover:text-accent-primary disabled:opacity-40 disabled:hover:text-ink-primary",
              inkClassName("primary"),
            )}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
