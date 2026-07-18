// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowIr, SceneIr } from "../packages/schema/src/ir";
import React, {
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import { renderDisplayList } from "../packages/runtime/src/backends/canvas/canvas-renderer";
import { hitTest } from "../packages/runtime/src/backends/canvas/hit-test";
import { SvgFallback } from "../packages/runtime/src/backends/svg/svg-fallback";
import type { DisplayList } from "../packages/runtime/src/display-list";
import { evaluateScene } from "../packages/runtime/src/evaluate/scene-evaluator";
import type {
  EvaluatedScene,
  SemanticProjection,
} from "../packages/runtime/src/evaluate/types";
import {
  beginExploration,
  resumeLesson,
  type ExplorationSnapshot,
} from "../packages/runtime/src/exploration";
import {
  NarratorController,
} from "../packages/runtime/src/narrative/narrator";
import type { KokoroNarratorSnapshot } from "../packages/runtime/src/narrative/kokoro-narrator";
import {
  SubtitleOverlay,
  type SubtitleState,
} from "../packages/runtime/src/narrative/subtitle-overlay";
import {
  evaluateNarrativeTimeline,
  type NarrativeCue,
} from "../packages/runtime/src/narrative/timeline";
import { TimelinePlayer } from "../packages/runtime/src/player";
import { SemanticTwin } from "../packages/runtime/src/semantic/semantic-twin";
import {
  createInitialSceneState,
  sceneReducer,
  type SceneState,
} from "../packages/runtime/src/store";

import { previewDurationMs, previewScene } from "./fixture";
import {
  createPreviewNarratorBackend,
  prewarmPreviewNarrator,
  subscribePreviewKokoroState,
  unlockPreviewSpeech,
} from "./narrator-backend";

type CanvasTool = "play" | "select";

/** Preview narrator modes: silent off, audible on, or muted cue tracking. */
export type NarratorMode = "off" | "on" | "muted";

type UnknownRecord = Readonly<Record<string, unknown>>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

/** Splits authored narration into clause-sized subtitle/speech units. */
export function splitNarrationClauses(narration: string): readonly string[] {
  const trimmed = narration.trim();
  if (trimmed === "") {
    return [];
  }
  const byComma = trimmed
    .split(/(?<=,)\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (byComma.length > 1) {
    return byComma;
  }
  return trimmed
    .split(/(?<=\.)\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/**
 * Narrow replaceable adapter: prefer `narrativeTrack`, else derive timed cues
 * from legacy `narration` paced to estimated speech duration so SpeechSynthesis
 * is not crushed into animation-only flash timings.
 */
export function sceneNarrativeCues(scene: SceneIr): readonly NarrativeCue[] {
  const track = scene.narrativeTrack;
  if (track !== undefined && track.cues.length > 0) {
    return track.cues.map((cue) =>
      Object.freeze({
        id: cue.id,
        atMs: cue.startMs,
        durationMs: Math.max(1, cue.endMs - cue.startMs),
        spokenText: cue.spokenText,
        subtitleText: cue.subtitleText,
      }),
    );
  }

  const narration = text(record(scene).narration).trim();
  if (narration === "") {
    return [];
  }

  const clauses = splitNarrationClauses(narration);
  const sceneEndMs = Math.max(1, previewDurationMs(scene));
  if (clauses.length === 0) {
    return [];
  }
  if (clauses.length === 1) {
    const spoken = clauses[0] ?? narration;
    return [
      Object.freeze({
        id: `${scene.id}:narration`,
        atMs: 0,
        durationMs: Math.max(sceneEndMs, estimateSpeechMs(spoken)),
        spokenText: spoken,
        subtitleText: spoken,
      }),
    ];
  }

  const speechDurations = clauses.map((clause) => estimateSpeechMs(clause));
  const totalSpeechMs = speechDurations.reduce(
    (sum, duration) => sum + duration,
    0,
  );
  const paceMs = Math.max(sceneEndMs, totalSpeechMs);
  const cues: NarrativeCue[] = [];
  let cursorMs = 0;
  for (let index = 0; index < clauses.length; index += 1) {
    const spokenText = clauses[index] ?? narration;
    const share = (speechDurations[index] ?? 0) / totalSpeechMs;
    const durationMs = Math.max(
      1,
      index + 1 === clauses.length
        ? paceMs - cursorMs
        : Math.round(paceMs * share),
    );
    cues.push(
      Object.freeze({
        id: `${scene.id}:narration-${index}`,
        atMs: cursorMs,
        durationMs,
        spokenText,
        subtitleText: spokenText,
      }),
    );
    cursorMs += durationMs;
  }
  return cues;
}

/** Rough spoken duration at ~150 wpm with a readable floor. */
function estimateSpeechMs(textValue: string): number {
  const words = textValue.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1_200, Math.round((words / 2.5) * 1_000));
}

export { unlockPreviewSpeech } from "./narrator-backend";

/** Builds a FlowIr host document from the live preview scene fixture. */
function previewFlow(): FlowIr {
  const scene = previewScene();
  return {
    irVersion: 1,
    id: "request-flow",
    title: "Request flow",
    capabilities: [],
    tokens: {},
    scenes: [scene],
    sourceMap: scene.sourceMap,
  };
}

function sceneTranscript(scene: SceneIr): string {
  return text(record(scene).narration, "No narration.");
}

/** True when a 2D canvas drawing context can be obtained in this environment. */
function canvas2dAvailable(): boolean {
  if (typeof document === "undefined") {
    return false;
  }
  try {
    const probe = document.createElement("canvas");
    return probe.getContext("2d") !== null;
  } catch {
    return false;
  }
}

function twinProjectionFromEvaluated(
  evaluated: EvaluatedScene,
  transcript: string,
): SemanticProjection {
  return {
    ...evaluated.semantic,
    captions: transcript === "" ? [] : [transcript],
  };
}

type CanvasStageProps = Readonly<{
  displayList: DisplayList;
  interactive: boolean;
  onSelectEntity(entityId: string): void;
}>;

function CanvasStage({
  displayList,
  interactive,
  onSelectEntity,
}: CanvasStageProps): ReactNode {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const context = canvas.getContext("2d");
    if (context === null) {
      return;
    }
    const { paintBounds } = displayList;
    const ratio =
      typeof window !== "undefined" && Number.isFinite(window.devicePixelRatio)
        ? Math.max(window.devicePixelRatio, 1)
        : 1;
    const width = Math.max(1, Math.ceil(paintBounds.width * ratio));
    const height = Math.max(1, Math.ceil(paintBounds.height * ratio));
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${Math.max(paintBounds.width, 1)}px`;
    canvas.style.height = `${Math.max(paintBounds.height, 1)}px`;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, width, height);
    renderDisplayList(context, displayList, { devicePixelRatio: ratio });
  }, [displayList]);

  function pointerToScene(event: ReactPointerEvent<HTMLCanvasElement>): {
    x: number;
    y: number;
  } | null {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return null;
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }
    const { paintBounds } = displayList;
    return {
      x:
        paintBounds.x +
        ((event.clientX - rect.left) / rect.width) * paintBounds.width,
      y:
        paintBounds.y +
        ((event.clientY - rect.top) / rect.height) * paintBounds.height,
    };
  }

  return (
    <canvas
      aria-hidden="true"
      className="aiperf-flow__canvas aiperf-flow__stage"
      data-backend="canvas"
      onPointerDown={(event) => {
        if (!interactive) {
          return;
        }
        const point = pointerToScene(event);
        if (point === null) {
          return;
        }
        const hit = hitTest(displayList.hitRegions, point);
        if (hit !== undefined) {
          onSelectEntity(hit.semanticId);
        }
      }}
      ref={canvasRef}
    />
  );
}

type CinematicStageProps = Readonly<{
  evaluated: EvaluatedScene;
  projection: SemanticProjection;
  preferCanvas: boolean;
  focusedEntityId: string | null;
  selectedEntityId: string | null;
  selectInteractive: boolean;
  subtitleState: SubtitleState;
  onSubtitlesEnabledChange(enabled: boolean): void;
  reducedMotion: boolean;
  onFocus(entityId: string): void;
  onActivate(entityId: string): void;
}>;

function CinematicStage({
  evaluated,
  projection,
  preferCanvas,
  focusedEntityId,
  selectedEntityId,
  selectInteractive,
  subtitleState,
  onSubtitlesEnabledChange,
  reducedMotion,
  onFocus,
  onActivate,
}: CinematicStageProps): ReactNode {
  return (
    <section
      aria-label="Scene stage"
      className="aiperf-flow__scene"
      data-backend={preferCanvas ? "canvas" : "svg"}
    >
      {preferCanvas ? (
        <CanvasStage
          displayList={evaluated.displayList}
          interactive={selectInteractive}
          onSelectEntity={onActivate}
        />
      ) : (
        <div className="aiperf-flow__stage" data-backend="svg">
          <SvgFallback
            displayList={evaluated.displayList}
            focusedEntityId={focusedEntityId}
            onFocusEntity={onFocus}
            onSelectEntity={onActivate}
            scene={evaluated}
            selectedEntityIds={
              selectedEntityId === null ? [] : [selectedEntityId]
            }
          />
        </div>
      )}
      <SemanticTwin
        compact
        focusedEntityId={focusedEntityId}
        onActivate={onActivate}
        onFocus={onFocus}
        projection={projection}
        selectedEntityId={selectedEntityId}
      />
      <SubtitleOverlay
        onEnabledChange={onSubtitlesEnabledChange}
        reducedMotion={reducedMotion}
        state={subtitleState}
      />
    </section>
  );
}

export function App() {
  const flow = useMemo(() => previewFlow(), []);
  const scenes = flow.scenes;
  const sourceName = text(flow.sourceMap.source, `${flow.id}.flow`);
  const preferCanvas = useMemo(() => canvas2dAvailable(), []);

  const [sceneIndex, setSceneIndex] = useState(0);
  const [timeMs, setTimeMs] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [browserCollapsed, setBrowserCollapsed] = useState(false);
  const [tool, setTool] = useState<CanvasTool>("play");
  const [exploration, setExploration] = useState<ExplorationSnapshot | null>(
    null,
  );
  const [focusedEntityId, setFocusedEntityId] = useState<string | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [narratorMode, setNarratorMode] = useState<NarratorMode>("on");
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(true);
  const [kokoroState, setKokoroState] = useState<KokoroNarratorSnapshot | null>(
    null,
  );
  const narratorBackend = useMemo(() => createPreviewNarratorBackend(), []);

  const scene = scenes[sceneIndex];
  const [sceneState, dispatch] = useReducer(
    sceneReducer,
    scene?.id ?? "unavailable",
    createInitialSceneState,
  );

  const timeline = scene === undefined ? [] : scene.timeline;
  const narrativeCues = useMemo(
    () => (scene === undefined ? [] : sceneNarrativeCues(scene)),
    [scene],
  );
  const player = useMemo(
    () =>
      new TimelinePlayer(timeline, undefined, (snapshot) => {
        setTimeMs(snapshot.timeMs);
        if (snapshot.complete) {
          setPlaying(false);
        }
      }),
    [timeline],
  );
  const narrator = useMemo(() => {
    const cues = narrativeCues.map((cue) => ({
      id: cue.id,
      atMs: cue.atMs,
      text: cue.spokenText,
    }));
    return new NarratorController(cues, narratorBackend);
  }, [narrativeCues, narratorBackend]);

  useEffect(() => {
    prewarmPreviewNarrator();
    return subscribePreviewKokoroState(setKokoroState);
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReducedMotion(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  useEffect(() => () => {
    player.pause();
    narrator.stop();
  }, [player, narrator]);

  useEffect(() => {
    setFocusedEntityId(null);
    setSelectedEntityId(null);
    setExploration(null);
    narrator.stop();
    dispatch({ type: "change-scene", sceneId: scene?.id ?? "unavailable" });
  }, [scene?.id, narrator]);

  useEffect(() => {
    if (exploration !== null) {
      player.pause();
      setPlaying(false);
      return;
    }
    if (reducedMotion) {
      player.seek(Number.POSITIVE_INFINITY);
      setPlaying(false);
      return;
    }
    if (playing) {
      player.play();
    } else {
      player.pause();
    }
  }, [exploration, playing, reducedMotion, player]);

  const durationMs = scene === undefined ? 0 : previewDurationMs(scene);
  const progress =
    durationMs <= 0 ? 0 : Math.min(100, (timeMs / durationMs) * 100);
  const transcript = scene === undefined ? "" : sceneTranscript(scene);
  const evaluationTimeMs =
    exploration === null
      ? Math.trunc(timeMs)
      : Math.trunc(exploration.authored.playbackTimeMs);
  const narrativeSnapshot = useMemo(
    () =>
      evaluateNarrativeTimeline(narrativeCues, evaluationTimeMs, {
        reducedMotion,
      }),
    [narrativeCues, evaluationTimeMs, reducedMotion],
  );
  const subtitleState = useMemo((): SubtitleState => {
    const active = narrativeSnapshot.activeCue;
    return {
      enabled: subtitlesEnabled,
      activeCue:
        active === null
          ? null
          : {
              id: active.id,
              text: narrativeSnapshot.subtitleText ?? active.spokenText,
              speaker: "Narrator",
            },
    };
  }, [narrativeSnapshot, subtitlesEnabled]);

  useEffect(() => {
    const beatMs = evaluationTimeMs;
    if (narratorMode === "off" || reducedMotion) {
      narrator.stop();
      return;
    }
    narrator.setMuted(narratorMode === "muted");
    if (exploration !== null || !playing) {
      narrator.pause(beatMs);
      return;
    }
    const status = narrator.snapshot().status;
    if (status === "paused") {
      narrator.resume(beatMs);
    } else if (status !== "playing") {
      narrator.play(beatMs);
    }
    narrator.sync(beatMs);
  }, [
    evaluationTimeMs,
    exploration,
    narrator,
    narratorMode,
    playing,
    reducedMotion,
  ]);

  const evaluating = useMemo((): EvaluatedScene | null => {
    if (scene === undefined || !Array.isArray(record(scene).roots)) {
      return null;
    }
    try {
      return evaluateScene(scene, evaluationTimeMs);
    } catch {
      return null;
    }
  }, [scene, evaluationTimeMs]);

  const twinProjection =
    evaluating === null
      ? null
      : twinProjectionFromEvaluated(evaluating, transcript);

  function navigate(nextIndex: number): void {
    narrator.stop();
    player.reset();
    setPlaying(false);
    setTimeMs(0);
    setExploration(null);
    setSceneIndex(nextIndex);
  }

  function cycleNarratorMode(): void {
    unlockPreviewSpeech();
    setNarratorMode((mode) =>
      mode === "on" ? "muted" : mode === "muted" ? "off" : "on",
    );
  }

  function togglePlayback(): void {
    if (exploration !== null) {
      return;
    }
    unlockPreviewSpeech();
    setTool("play");
    setPlaying((value) => !value);
  }

  function startExploration(fromState: SceneState): void {
    player.pause();
    setPlaying(false);
    const snapshot = beginExploration({
      ...fromState,
      playbackTimeMs: timeMs,
      playbackStatus: "paused",
    });
    setExploration(snapshot);
    dispatch({ type: "set-camera-takeover", active: true });
    dispatch({
      type: "set-playback",
      timeMs: snapshot.authored.playbackTimeMs,
      status: "paused",
    });
  }

  function resumeFromExploration(): void {
    if (exploration === null) {
      return;
    }
    const restored = resumeLesson(exploration);
    setExploration(null);
    dispatch({ type: "set-camera-takeover", active: false });
    dispatch({
      type: "set-playback",
      timeMs: restored.playbackTimeMs,
      status: restored.playbackStatus,
    });
    player.seek(restored.playbackTimeMs);
    setTimeMs(restored.playbackTimeMs);
    setTool("play");
    if (restored.playbackStatus === "playing" && !reducedMotion) {
      setPlaying(true);
    }
  }

  function focusEntity(entityId: string): void {
    setFocusedEntityId(entityId);
  }

  function activateEntity(entityId: string): void {
    setFocusedEntityId(entityId);
    setSelectedEntityId(entityId);
    dispatch({ type: "open-inspector", nodeId: entityId });
    if (tool === "select" || exploration !== null) {
      if (exploration === null) {
        startExploration({
          ...sceneState,
          selectedNodeId: entityId,
          playbackTimeMs: timeMs,
          playbackStatus: playing ? "playing" : "paused",
        });
      } else {
        setExploration(
          beginExploration({
            ...exploration.authored,
            selectedNodeId: entityId,
            playbackTimeMs: exploration.authored.playbackTimeMs,
            playbackStatus: "paused",
            temporaryCameraTakeover: true,
          }),
        );
      }
      return;
    }
    // Viewer interaction pauses at the current beat by default.
    if (playing) {
      startExploration({
        ...sceneState,
        selectedNodeId: entityId,
        playbackTimeMs: timeMs,
        playbackStatus: "playing",
      });
      setTool("select");
    }
  }

  const exploring = exploration !== null;
  const sceneTitle = text(scene?.title, "Untitled scene");
  const sceneSummary = text(scene?.summary, "");
  const chapterLabels = scenes.map((entry) => text(entry.title, entry.id));

  return (
    <div className="preview-shell">
      <header className="preview-topbar">
        <div className="preview-brand-cluster">
          <button
            aria-expanded={!browserCollapsed}
            aria-label={
              browserCollapsed ? "Open Flow browser" : "Close Flow browser"
            }
            className="flow-browser-trigger"
            onClick={() => setBrowserCollapsed((value) => !value)}
            type="button"
          >
            <span />
            <span />
            <span />
          </button>
          <a className="preview-brand" href="/" aria-label="AIPerf Flow home">
            <span className="preview-brand-mark" aria-hidden="true">
              F
            </span>
            <span>
              <strong>AIPerf Flow</strong>
              <small>Semantic runtime</small>
            </span>
          </a>
        </div>
        <div className="preview-status">
          <span className="preview-status-dot" />
          Runtime connected
          <span className="preview-build">cinematic / live</span>
        </div>
      </header>

      <div
        className="flow-workspace"
        data-browser-collapsed={browserCollapsed ? "true" : "false"}
      >
        <aside className="flow-browser" aria-label="Flow browser">
          <div className="flow-browser-head">
            <div>
              <span className="flow-browser-kicker">Workspace</span>
              <strong>Flows</strong>
            </div>
            <button
              aria-label="Collapse Flow browser"
              onClick={() => setBrowserCollapsed(true)}
              type="button"
            >
              ‹
            </button>
          </div>

          <label className="flow-browser-search">
            <span aria-hidden="true">⌕</span>
            <input
              aria-label="Search flows"
              placeholder="Find a flow or scene"
              type="search"
            />
            <kbd>⌘K</kbd>
          </label>

          <nav className="flow-tree" aria-label="Flow files and scenes">
            <details open>
              <summary>
                <span className="flow-file-mark" aria-hidden="true">
                  ◆
                </span>
                <span>{sourceName}</span>
                <small>{scenes.length}</small>
              </summary>
              <div className="flow-tree-branch">
                <section>
                  <p>{flow.title}</p>
                  <ul>
                    {scenes.map((entry, index) => {
                      const active = index === sceneIndex;
                      return (
                        <li key={entry.id}>
                          <button
                            aria-current={active ? "page" : undefined}
                            className={active ? "is-active" : undefined}
                            onClick={() => navigate(index)}
                            type="button"
                          >
                            <span aria-hidden="true">
                              {active ? "●" : "○"}
                            </span>
                            {text(entry.title, entry.id)}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </section>
              </div>
            </details>
          </nav>

          <footer className="flow-browser-foot">
            <span className="preview-status-dot" />
            1 flow · {scenes.length} scene{scenes.length === 1 ? "" : "s"}
          </footer>
        </aside>

        <main className="runtime-story">
          <div className="story-rail-scroll">
            <ol className="story-rail" aria-label="Flow chapters">
              {chapterLabels.map((label, index) => (
                <li className="story-rail-item" key={`${label}-${index}`}>
                  <button
                    aria-current={index === sceneIndex ? "step" : undefined}
                    className="story-rail-dot"
                    data-accent="transport"
                    data-state={
                      index === sceneIndex
                        ? "active"
                        : index < sceneIndex
                          ? "done"
                          : "upcoming"
                    }
                    onClick={() => navigate(index)}
                    type="button"
                  >
                    <span className="story-rail-index">{index + 1}</span>
                    <span className="story-rail-label">{label}</span>
                  </button>
                </li>
              ))}
            </ol>
          </div>

          <section
            className="story-stage"
            data-accent="transport"
            aria-label="Rendered Flow scene"
          >
            <article className="story-stage-header">
              <p className="story-kicker">
                <span className="story-kicker-chapter">
                  {String(sceneIndex + 1).padStart(2, "0")} /{" "}
                  {String(scenes.length).padStart(2, "0")}
                </span>
                <span className="story-kicker-text">{flow.title}</span>
              </p>
              <div className="story-heading-line">
                <h1 className="story-title">{sceneTitle}</h1>
                <p className="story-blurb">{sceneSummary}</p>
              </div>
              <p className="story-brief-evidence">{transcript}</p>
            </article>

            <div className="story-figure">
              <div className="preview-canvas-tools" aria-label="Canvas controls">
                <button
                  type="button"
                  aria-label={
                    narratorMode === "on"
                      ? "Mute narrator"
                      : narratorMode === "muted"
                        ? "Turn narrator off"
                        : "Turn narrator on"
                  }
                  aria-pressed={narratorMode !== "off"}
                  className={narratorMode !== "off" ? "is-active" : undefined}
                  data-narrator-mode={narratorMode}
                  onClick={cycleNarratorMode}
                >
                  {narratorMode === "on"
                    ? "🔊"
                    : narratorMode === "muted"
                      ? "🔇"
                      : "⏹"}
                </button>
                <button
                  type="button"
                  aria-label={playing && !exploring ? "Pause scene" : "Play scene"}
                  className={
                    tool === "play" && !exploring ? "is-active" : undefined
                  }
                  disabled={exploring}
                  onClick={togglePlayback}
                >
                  {playing && !exploring ? "Ⅱ" : "▶"}
                </button>
                <button
                  type="button"
                  aria-label="Select tool"
                  className={
                    tool === "select" || exploring ? "is-active" : undefined
                  }
                  onClick={() => {
                    setTool("select");
                    if (!exploring && playing) {
                      startExploration({
                        ...sceneState,
                        playbackTimeMs: timeMs,
                        playbackStatus: "playing",
                      });
                    }
                  }}
                >
                  ↖
                </button>
                {exploring ? (
                  <button
                    type="button"
                    aria-label="Resume lesson from current beat"
                    className="is-active"
                    onClick={resumeFromExploration}
                  >
                    ↺
                  </button>
                ) : null}
              </div>
              <div className="story-scene-progress" aria-hidden="true">
                <span style={{ width: `${progress}%` }} />
              </div>
              {evaluating !== null && twinProjection !== null ? (
                <CinematicStage
                  evaluated={evaluating}
                  focusedEntityId={focusedEntityId}
                  onActivate={activateEntity}
                  onFocus={focusEntity}
                  onSubtitlesEnabledChange={setSubtitlesEnabled}
                  preferCanvas={preferCanvas}
                  projection={twinProjection}
                  reducedMotion={reducedMotion}
                  selectInteractive={tool === "select" || exploring}
                  selectedEntityId={selectedEntityId}
                  subtitleState={subtitleState}
                />
              ) : (
                <section
                  aria-label="Scene fallback"
                  className="aiperf-flow__scene-fallback"
                  role="status"
                >
                  <p>
                    {text(
                      scene?.summary,
                      "This scene could not be displayed.",
                    )}
                  </p>
                  <p>
                    {text(
                      scene?.fallback,
                      "Use the transcript for this scene.",
                    )}
                  </p>
                </section>
              )}
            </div>
          </section>

          <nav className="story-controls" aria-label="Flow navigation">
            <button
              className="story-nav-button"
              type="button"
              disabled={sceneIndex === 0 && timeMs === 0 && !exploring}
              onClick={() => {
                if (exploring) {
                  resumeFromExploration();
                }
                if (sceneIndex > 0) {
                  navigate(sceneIndex - 1);
                  return;
                }
                narrator.stop();
                player.reset();
                setTimeMs(0);
                setPlaying(!reducedMotion);
                setExploration(null);
                dispatch({ type: "set-camera-takeover", active: false });
              }}
            >
              <span aria-hidden="true">&larr;</span> Back
            </button>
            <p className="story-progress" role="status">
              Scene {sceneIndex + 1} of {scenes.length} ·{" "}
              {(timeMs / 1000).toFixed(1)}s
              {exploring ? " · exploring" : ""}
              {sceneState.temporaryCameraTakeover ? " · camera takeover" : ""}
              {narratorMode === "muted"
                ? " · narrator muted"
                : narratorMode === "off"
                  ? " · narrator off"
                  : ""}
              {subtitlesEnabled && narrativeSnapshot.subtitleText !== null
                ? " · subtitled"
                : ""}
              {kokoroState?.status === "loading"
                ? ` · voice ${Math.round((kokoroState.progress ?? 0) * 100)}%`
                : kokoroState?.status === "needs-user-activation"
                  ? " · tap play for voice"
                  : kokoroState?.engine === "webgpu"
                    ? " · kokoro gpu"
                    : kokoroState?.engine === "wasm"
                      ? " · kokoro wasm"
                      : ""}
            </p>
            <button
              className="story-nav-button story-nav-button-primary"
              type="button"
              disabled={sceneIndex >= scenes.length - 1}
              onClick={() => navigate(sceneIndex + 1)}
            >
              Next <span aria-hidden="true">&rarr;</span>
            </button>
          </nav>
        </main>
      </div>
    </div>
  );
}
