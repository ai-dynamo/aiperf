/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { usePrefersReducedMotion } from "./diagram/usePrefersReducedMotion";
import {
  DEFAULT_PLAYBACK_SPEED,
  isPlaybackSpeed,
  narrationSupported,
  PLAYBACK_SPEEDS,
  stopNarration,
  unlockSpeech,
  type PlaybackSpeed,
} from "./narration";
import type { DeckDefinition, MentalModelProps, SlideDefinition } from "./types";
import { slideNarrations } from "./types";
import {
  formatSlideDuration,
  formatSlideshowDuration,
  useTimedSlideshow,
} from "./useTimedSlideshow";
import { useSpeechVoices } from "./useSpeechVoices";
import { useIdleChrome, usePresentMode } from "./usePresentationMode";
import {
  BrandMark,
  Button,
  Pill,
  StartGate,
  Subtitles,
  VoicePicker,
  useCanvasState,
} from "./ui";

function formatSpeedLabel(speed: PlaybackSpeed): string {
  return `${speed}×`;
}

function stepLabel(slide: SlideDefinition): string {
  const raw = (slide.eyebrow || slide.title || "").trim();
  if (raw.length <= 20) return raw;
  return `${raw.slice(0, 18)}…`;
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

/**
 * Packages-only playback host: voice (useTimedSlideshow) and SceneRenderer timelines
 * share `restartKey` so slide changes, revisits, play, and restart stay in sync.
 * FinalCard mounts only when the DeckPackage authored one — never a last-slide clone.
 */
export function ExplainerShell({ deck }: { deck: DeckDefinition }) {
  const location = useLocation();
  const { slides, storagePrefix, classPrefix, eyebrowLabel, startGateTitle } = deck;
  const narrations = useMemo(() => slideNarrations(slides), [slides]);

  // Step and playing are session-only so a refresh always opens on slide 1 / start gate.
  const [stored, setStored] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [narrationEnabled, setNarrationEnabled] = useCanvasState<boolean>(
    storagePrefix,
    "narration",
    true,
  );
  const [voiceURI, setVoiceURI] = useCanvasState<string>(storagePrefix, "voice", "");
  const [speedRaw, setSpeedRaw] = useCanvasState<number>(
    storagePrefix,
    "speed",
    DEFAULT_PLAYBACK_SPEED,
  );
  const speed: PlaybackSpeed = isPlaybackSpeed(speedRaw)
    ? speedRaw
    : DEFAULT_PLAYBACK_SPEED;
  const [started, setStarted] = useState(false);
  const [restartKey, setRestartKey] = useState(0);
  const [notesOpen, setNotesOpen] = useState(false);
  const shellRef = useRef<HTMLDivElement>(null);
  const { presenting, togglePresent } = usePresentMode(shellRef);
  const { chromeVisible, revealChrome } = useIdleChrome(started, notesOpen);
  const reducedMotion = usePrefersReducedMotion();
  const voices = useSpeechVoices();
  const index = Number.isInteger(stored) && stored >= 0 && stored < slides.length ? stored : 0;
  const slide = slides[index];
  const speechAvailable = narrationSupported();
  const FinalCard = deck.FinalCard;
  const MentalModel = deck.MentalModel as (props: MentalModelProps) => ReactNode;

  const bumpRestart = () => setRestartKey((key) => key + 1);

  useEffect(() => {
    stopNarration();
    setPlaying(false);
  }, [location.pathname, setPlaying]);

  const advance = () => {
    if (!playing) return;
    if (index === slides.length - 1) {
      stopNarration();
      setPlaying(false);
      return;
    }
    setStored(index + 1);
    bumpRestart();
  };

  const { activeWordIndex } = useTimedSlideshow({
    index,
    playing: started && playing,
    narrationEnabled,
    voiceURI,
    narrations,
    restartKey,
    speed,
    onAdvance: advance,
  });

  const setSpeed = (next: PlaybackSpeed) => {
    setSpeedRaw(next);
  };

  const goTo = (next: number) => {
    const nextIndex = Math.max(0, Math.min(slides.length - 1, next));
    stopNarration();
    setStored(nextIndex);
    if (started) {
      setPlaying(true);
      bumpRestart();
    }
  };

  const begin = (withNarration: boolean) => {
    if (withNarration) unlockSpeech();
    setNarrationEnabled(withNarration);
    setStarted(true);
    setPlaying(true);
    bumpRestart();
    revealChrome();
  };

  const togglePlayback = () => {
    if (playing) {
      stopNarration();
      setPlaying(false);
      return;
    }
    unlockSpeech();
    const nextIndex = index === slides.length - 1 ? 0 : index;
    if (nextIndex !== index) setStored(nextIndex);
    setStarted(true);
    setPlaying(true);
    bumpRestart();
  };

  const onKeyDown = (event: {
    key: string;
    target: EventTarget | null;
    preventDefault: () => void;
  }) => {
    const el = event.target as HTMLElement | null;
    if (el?.matches("input, textarea, select, button, [role='button']")) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      goTo(index - 1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      goTo(index + 1);
    }
    if (event.key.toLowerCase() === "f") {
      event.preventDefault();
      void togglePresent();
    }
    if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  };

  return (
    <div
      ref={shellRef}
      className={[
        "ex-page",
        "ex-shell",
        presenting ? "ex-shell--present" : "",
        started && !chromeVisible ? "ex-shell--chrome-hidden" : "",
        `${classPrefix}-page`,
      ]
        .filter(Boolean)
        .join(" ")}
      tabIndex={0}
      onKeyDown={onKeyDown}
      onPointerDown={revealChrome}
    >
      <style>{deck.css}</style>
      {!started ? (
        <StartGate
          title={startGateTitle}
          speechAvailable={speechAvailable}
          voices={voices}
          selectedVoiceURI={voiceURI}
          onVoiceSelect={setVoiceURI}
          onStartWithNarration={() => begin(true)}
          onStartSilent={() => begin(false)}
        />
      ) : null}

      <header className="ex-topbar ex-chrome ex-chrome--top">
        <div className="ex-topbar__brand">
          <BrandMark />
          <Link to="/" className="ex-link ex-link--muted">
            ← Home
          </Link>
          <span className="ex-meta">AIPerf · {eyebrowLabel}</span>
        </div>
        <div className="ex-topbar__controls">
          <Button
            className="ex-btn--ghost"
            variant="secondary"
            onClick={togglePlayback}
          >
            {playing ? "Pause" : index === slides.length - 1 ? "Replay" : "Play"}
          </Button>
          <Button
            className="ex-btn--ghost"
            variant="secondary"
            onClick={() => {
              if (!speechAvailable) return;
              if (narrationEnabled) {
                stopNarration();
                setNarrationEnabled(false);
                return;
              }
              unlockSpeech();
              setNarrationEnabled(true);
            }}
          >
            {!speechAvailable ? "No audio" : narrationEnabled ? "Mute" : "Unmute"}
          </Button>
          <div className="ex-speed-group" role="group" aria-label="Playback speed">
            <div className="ex-pill-group">
              {PLAYBACK_SPEEDS.map((preset) => (
                <Pill
                  key={preset}
                  size="sm"
                  active={preset === speed}
                  title={`Play at ${formatSpeedLabel(preset)}`}
                  onClick={() => setSpeed(preset)}
                >
                  {formatSpeedLabel(preset)}
                </Pill>
              ))}
            </div>
          </div>
          <Button
            className="ex-btn--present"
            variant="secondary"
            onClick={() => void togglePresent()}
          >
            {presenting ? "Exit present" : "Present"}
          </Button>
        </div>
      </header>

      <nav className="ex-progress ex-chrome ex-chrome--progress" aria-label="Slide steps">
        {slides.map((entry, i) => {
          const state = i < index ? "done" : i === index ? "current" : "todo";
          return (
            <button
              type="button"
              key={`${entry.title}-${i}`}
              className={`ex-progress__segment ex-progress__segment--${state}`}
              title={`${i + 1}. ${stepLabel(entry)}`}
              aria-label={`Go to slide ${i + 1}: ${entry.title}`}
              aria-current={i === index ? "step" : undefined}
              onClick={() => goTo(i)}
            >
              <span className="ex-progress__fill" />
            </button>
          );
        })}
      </nav>

      <main
        key={`slide-${index}-${restartKey}`}
        className={`ex-content-card ex-cinematic-stage ${classPrefix}-slide`}
        hidden={index === slides.length - 1 && FinalCard !== undefined}
        aria-hidden={index === slides.length - 1 && FinalCard !== undefined}
      >
        <div className={`ex-content-card__diagram ex-stage-hero ${classPrefix}-stage`}>
          <MentalModel
            slideIndex={index}
            slide={slide}
            playing={started && playing}
            restartKey={restartKey}
            reducedMotion={reducedMotion}
            playbackRate={speed}
          />
        </div>

        <div className="ex-stage-footer">
          {started ? (
            <div className="ex-subtitles-row" data-testid="ex-subtitles-row">
              <Subtitles
                text={slide.narration}
                activeWordIndex={activeWordIndex}
                visible
              />
            </div>
          ) : null}

          <section className={`ex-content-card__copy ex-stage-copy ${classPrefix}-hero`}>
            <div className="ex-eyebrow ex-eyebrow--accent">
              {pad2(index + 1)} / {pad2(slides.length)} · {slide.eyebrow}
            </div>
            <h1 className="ex-slide-title">{slide.title}</h1>
            <p className={`${classPrefix}-lede ex-lede`} style={{ margin: 0 }}>
              {slide.lede}
            </p>
          </section>
        </div>
      </main>

      {index === slides.length - 1 && FinalCard ? (
        <FinalCard
          playing={started && playing}
          restartKey={restartKey}
          reducedMotion={reducedMotion}
          playbackRate={speed}
        />
      ) : null}

      <div className="ex-bottom-nav ex-chrome ex-chrome--bottom">
        <div className="ex-bottom-nav__back">
          <Button variant="secondary" disabled={index === 0} onClick={() => goTo(index - 1)}>
            ← Back
          </Button>
        </div>
        <div className="ex-bottom-nav__chapter">
          Chapter {index + 1} of {slides.length}
        </div>
        <div className="ex-bottom-nav__next">
          <Button
            variant="secondary"
            disabled={index === slides.length - 1}
            onClick={() => goTo(index + 1)}
          >
            Next →
          </Button>
        </div>
      </div>

      <Button
        className="ex-notes-toggle"
        variant="secondary"
        onClick={() => setNotesOpen((open) => !open)}
      >
        {notesOpen ? "Close notes" : "Speaker notes"}
      </Button>

      {notesOpen ? (
        <aside className="ex-speaker-notes" aria-label="Speaker notes">
          <div className="ex-speaker-notes__header">
            <div>
              <div className="ex-eyebrow ex-eyebrow--accent">
                Slide {pad2(index + 1)} notes
              </div>
              <div className="ex-speaker-notes__title">{slide.title}</div>
            </div>
            <Button variant="secondary" onClick={() => setNotesOpen(false)}>
              Close
            </Button>
          </div>
          <div className="ex-speaker-notes__body">
            {slide.term ? (
              <div className="ex-term">
                <div className="ex-term__word">{slide.term.word}</div>
                <div className="ex-term__meaning">{slide.term.meaning}</div>
              </div>
            ) : null}
            <div className={`ex-points ${classPrefix}-points`}>
              {slide.points.map((point) => (
                <div key={point} className={`ex-point ${classPrefix}-point`}>
                  <span className="ex-point__mark">·</span>
                  <span>{point}</span>
                </div>
              ))}
            </div>
            {slide.caption ? <p className="ex-speaker-notes__caption">{slide.caption}</p> : null}
            <details className="ex-more">
              <summary>Voice &amp; timing</summary>
              <div className="ex-more__body">
                <VoicePicker
                  voices={voices}
                  selectedVoiceURI={voiceURI}
                  speechAvailable={speechAvailable}
                  onVoiceSelect={(next) => {
                    unlockSpeech();
                    setVoiceURI(next);
                    if (started && !narrationEnabled) setNarrationEnabled(true);
                  }}
                />
                <div className="ex-duration">
                  Slide ~{formatSlideDuration(slide.narration, speed)} · total{" "}
                  {formatSlideshowDuration(narrations, speed)}
                </div>
                <Button
                  variant="secondary"
                  onClick={() => {
                    unlockSpeech();
                    stopNarration();
                    setStored(0);
                    setStarted(true);
                    setPlaying(true);
                    bumpRestart();
                  }}
                >
                  Restart from beginning
                </Button>
              </div>
            </details>
          </div>
        </aside>
      ) : null}
    </div>
  );
}
