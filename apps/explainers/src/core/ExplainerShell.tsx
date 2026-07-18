/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { usePrefersReducedMotion } from "./diagram/usePrefersReducedMotion";
import { narrationSupported, stopNarration, unlockSpeech } from "./narration";
import type { DeckDefinition, MentalModelProps } from "./types";
import { slideNarrations } from "./types";
import {
  formatSlideDuration,
  formatSlideshowDuration,
  useTimedSlideshow,
} from "./useTimedSlideshow";
import { useSpeechVoices } from "./useSpeechVoices";
import {
  Button,
  Divider,
  Pill,
  Row,
  Stack,
  StartGate,
  Subtitles,
  Text,
  VoicePicker,
  useCanvasState,
  useHostTheme,
} from "./ui";

/**
 * Packages-only playback host: voice (useTimedSlideshow) and SceneRenderer timelines
 * share `restartKey` so slide changes, revisits, play, and restart stay in sync.
 * FinalCard mounts only when the DeckPackage authored one — never a last-slide clone.
 */
export function ExplainerShell({ deck }: { deck: DeckDefinition }) {
  const t = useHostTheme();
  const location = useLocation();
  const { slides, storagePrefix, classPrefix, eyebrowLabel, startGateTitle } = deck;
  const narrations = useMemo(() => slideNarrations(slides), [slides]);

  const [stored, setStored] = useCanvasState<number>(storagePrefix, "step", 0);
  const [playing, setPlaying] = useCanvasState<boolean>(storagePrefix, "playing", false);
  const [narrationEnabled, setNarrationEnabled] = useCanvasState<boolean>(
    storagePrefix,
    "narration",
    true,
  );
  const [voiceURI, setVoiceURI] = useCanvasState<string>(storagePrefix, "voice", "");
  const [started, setStarted] = useState(false);
  const [restartKey, setRestartKey] = useState(0);
  const reducedMotion = usePrefersReducedMotion();
  const voices = useSpeechVoices();
  const index = Number.isInteger(stored) && stored >= 0 && stored < slides.length ? stored : 0;
  const slide = slides[index];
  const speechAvailable = narrationSupported();
  const FinalCard = deck.FinalCard;
  // Package MentalModel forwards playing/restartKey/reducedMotion to SceneRenderer.
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
    // Keep playing so voice continues; bump restartKey so SceneRenderer
    // timelines restart for the new slide (same contract as goTo).
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
    onAdvance: advance,
  });

  const goTo = (next: number) => {
    const nextIndex = Math.max(0, Math.min(slides.length - 1, next));
    stopNarration();
    setStored(nextIndex);
    // Revisiting a slide (Back, pill, ←) restarts narration and SVG motion from the top.
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
    // Align SceneRenderer timeline start with the first narration utterance.
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
  };

  return (
    <div
      className={`${classPrefix}-page`}
      tabIndex={0}
      onKeyDown={onKeyDown}
      style={{ padding: 22, background: t.bg.editor }}
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
      <Stack gap={18}>
        <div className={`${classPrefix}-hero`}>
          <Stack gap={5}>
            <Row gap={10} align="center" wrap>
              <Link
                to="/"
                style={{
                  color: t.category.green,
                  textDecoration: "none",
                  fontSize: 14,
                  fontWeight: 650,
                }}
              >
                ← Explainers home
              </Link>
              <Text tone="secondary">·</Text>
              <div
                style={{
                  color: t.text.secondary,
                  fontSize: 14,
                  fontWeight: 650,
                  letterSpacing: "0.04em",
                }}
              >
                AIPERF · {eyebrowLabel} · STEP {index + 1} OF {slides.length}
              </div>
            </Row>
            <div>
              <div style={{ color: t.category.green, fontSize: 15, fontWeight: 700 }}>
                {slide.eyebrow}
              </div>
              <div
                style={{
                  fontSize: 24,
                  fontWeight: 700,
                  lineHeight: 1.2,
                  color: t.text.primary,
                  marginTop: 2,
                }}
              >
                {slide.title}
              </div>
            </div>
            <div className={`${classPrefix}-lede`} style={{ color: t.text.secondary }}>
              {slide.lede}
            </div>
          </Stack>
        </div>

        <div className={`${classPrefix}-rail`}>
          {slides.map((entry, i) => (
            <span key={entry.title} style={{ display: "contents" }}>
              <Pill active={i === index} onClick={() => goTo(i)} title={entry.title}>
                {String(i + 1)}
              </Pill>
            </span>
          ))}
        </div>

        <Row gap={10} align="center">
          <Button variant="secondary" disabled={index === 0} onClick={() => goTo(index - 1)}>
            Back
          </Button>
          <Text tone="secondary">
            {index + 1} / {slides.length}
          </Text>
          <Button
            disabled={index === slides.length - 1}
            style={{ background: t.category.green, color: t.text.onAccent }}
            onClick={() => goTo(index + 1)}
          >
            Next
          </Button>
        </Row>

        <Row gap={10} align="center" wrap>
          <Button
            style={{ background: t.category.green, color: t.text.onAccent }}
            onClick={() => {
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
              // Voice restarts with playing; bump so timelines restart in lockstep.
              bumpRestart();
            }}
          >
            {playing
              ? "Pause slideshow"
              : index === slides.length - 1
                ? "Replay slideshow"
                : "Play slideshow"}
          </Button>
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
            Restart
          </Button>
          <Button
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
            {!speechAvailable
              ? "Narration unavailable"
              : narrationEnabled
                ? "Mute narration"
                : "Enable narration"}
          </Button>
          <Text tone="secondary">
            Slide ~{formatSlideDuration(slide.narration)} · total{" "}
            {formatSlideshowDuration(narrations)}
          </Text>
        </Row>

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

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            width: "100%",
          }}
        >
          <div style={{ width: "100%", maxWidth: 720 }}>
            <Subtitles
              text={slide.narration}
              activeWordIndex={activeWordIndex}
              visible={started}
            />
          </div>
        </div>

        <Divider />

        <div key={`slide-${index}-${restartKey}`} className={`${classPrefix}-stage ${classPrefix}-slide`}>
          <MentalModel
            slideIndex={index}
            slide={slide}
            playing={started && playing}
            restartKey={restartKey}
            reducedMotion={reducedMotion}
          />
          <div className={`${classPrefix}-details`}>
            {slide.term ? (
              <div
                style={{
                  padding: 16,
                  borderRadius: 8,
                  background: t.fill.quaternary,
                  border: `1px solid ${t.stroke.secondary}`,
                }}
              >
                <div
                  style={{
                    color: t.category.green,
                    fontSize: 15,
                    fontWeight: 700,
                    marginBottom: 7,
                  }}
                >
                  {slide.term.word}
                </div>
                <div style={{ color: t.text.primary, fontSize: 16, lineHeight: 1.55 }}>
                  {slide.term.meaning}
                </div>
              </div>
            ) : null}
            <div style={{ borderTop: `2px solid ${t.stroke.primary}`, paddingTop: 14 }}>
              <div
                style={{ color: t.text.primary, fontSize: 16, fontWeight: 700, marginBottom: 12 }}
              >
                What happens here
              </div>
              <div className={`${classPrefix}-points`}>
                {slide.points.map((point) => (
                  <div key={point} className={`${classPrefix}-point`}>
                    <span style={{ color: t.category.green, fontWeight: 700 }}>·</span>
                    <span style={{ color: t.text.primary }}>{point}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {index === slides.length - 1 && FinalCard ? <FinalCard /> : null}

        <Divider />
        <Row gap={10} align="center">
          <Link
            to="/"
            style={{ color: t.text.secondary, textDecoration: "none", fontSize: 14, fontWeight: 600 }}
          >
            Explainers home
          </Link>
          <Text tone="secondary">·</Text>
          <Text tone="secondary">Use ← and → to move between steps.</Text>
        </Row>
      </Stack>
    </div>
  );
}
