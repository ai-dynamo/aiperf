// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import {
  Component,
  type ErrorInfo,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { renderDisplayList } from "./backends/canvas/canvas-renderer.js";
import { hitTest } from "./backends/canvas/hit-test.js";
import { SvgFallback } from "./backends/svg/svg-fallback.js";
import {
  beginCameraTakeover,
  type CameraTakeover,
  fitCameraTakeover,
  panCameraTakeover,
  resumeAuthoredCamera,
  zoomCameraTakeover,
} from "./camera-policy.js";
import { buildDisplayList, type DisplayList } from "./display-list.js";
import { evaluateScene } from "./evaluate/scene-evaluator.js";
import type {
  EvaluatedScene,
  SemanticEntityProjection,
  SemanticProjection,
} from "./evaluate/types.js";
import {
  beginExploration,
  type ExplorationSnapshot,
  resumeLesson,
  updateExploration,
} from "./exploration.js";
import {
  createBrowserSpeechSynthesisBackend,
  NarratorController,
  type NarratorBackend,
} from "./narrative/narrator.js";
import { sceneNarrativeCues } from "./narrative/scene-cues.js";
import {
  SubtitleOverlay,
  type SubtitleState,
} from "./narrative/subtitle-overlay.js";
import { evaluateNarrativeTimeline } from "./narrative/timeline.js";
import { type Clock, PerformanceClock, TimelinePlayer } from "./player.js";
import { createFoundationRegistry } from "./renderer.js";
import type { CapabilityRegistry } from "./registry.js";
import {
  createFocusCoordinator,
  type FocusCoordinator,
} from "./semantic/focus-coordinator.js";
import { SemanticTwin } from "./semantic/semantic-twin.js";
import { createInitialSceneState } from "./store.js";

type UnknownRecord = Readonly<Record<string, unknown>>;

const unavailableNarratorBackend: NarratorBackend = Object.freeze({
  available: false,
  voices: () => Object.freeze([]),
  speak: () => undefined,
  pause: () => undefined,
  resume: () => undefined,
  cancel: () => undefined,
});

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function sceneTranscript(scene: SceneIr): string {
  return text(record(scene).narration);
}

function requiredCapabilityIds(flow: FlowIr): readonly string[] {
  const requirements = record(flow).capabilities;
  return Array.isArray(requirements)
    ? requirements.map((requirement) => text(record(requirement).id)).filter(Boolean)
    : [];
}

function unavailableCapabilities(
  flow: FlowIr,
  registry: CapabilityRegistry,
): readonly string[] {
  return requiredCapabilityIds(flow).filter((id) => {
    try {
      registry.require(id);
      return false;
    } catch {
      return true;
    }
  });
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

/** Ensures an evaluated scene always carries a display list for backends. */
function withDisplayList(evaluated: EvaluatedScene): EvaluatedScene {
  if (evaluated.displayList !== undefined) {
    return evaluated;
  }
  return {
    ...evaluated,
    displayList: buildDisplayList({
      commands: [],
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 1, height: 1 },
      damageBounds: { x: 0, y: 0, width: 1, height: 1 },
    }),
  };
}

/** Projects the active subtitle cue into evaluated scene semantics. */
function twinProjectionFromEvaluated(
  evaluated: EvaluatedScene,
  subtitle: SubtitleState,
): SemanticProjection {
  const active = subtitle.enabled ? subtitle.activeCue : null;
  return {
    ...evaluated.semantic,
    ...(active === null
      ? { captions: [] }
      : {
          transcriptCueId: active.id,
          captions: [active.text],
        }),
  };
}

function entityFromProjection(
  projection: SemanticProjection | null,
  entityId: string | null,
): SemanticEntityProjection | undefined {
  if (projection === null || entityId === null) {
    return undefined;
  }
  return projection.entities.find((entity) => entity.id === entityId);
}

type SceneFailureProps = Readonly<{
  scene: SceneIr;
  reason?: string;
}>;

function SceneFailure({ scene, reason }: SceneFailureProps): ReactNode {
  const properties = record(scene);
  return (
    <section
      aria-label="Scene fallback"
      className="aiperf-flow__scene-fallback"
      role="status"
    >
      <p>{text(properties.summary, "This scene could not be displayed.")}</p>
      <p>{text(properties.fallback, "Use the transcript for this scene.")}</p>
      {reason === undefined ? null : (
        <p className="aiperf-flow__error-detail">{reason}</p>
      )}
    </section>
  );
}

type SceneErrorBoundaryProps = Readonly<{
  scene: SceneIr;
  children: ReactNode;
}>;

type SceneErrorBoundaryState = Readonly<{ error: Error | null }>;

class SceneErrorBoundary extends Component<
  SceneErrorBoundaryProps,
  SceneErrorBoundaryState
> {
  override state: SceneErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): SceneErrorBoundaryState {
    return { error };
  }

  override componentDidCatch(_error: Error, _info: ErrorInfo): void {
    // Rendering errors remain isolated to the scene; navigation and transcript stay mounted.
  }

  override render(): ReactNode {
    return this.state.error === null ? (
      this.props.children
    ) : (
      <SceneFailure reason={this.state.error.message} scene={this.props.scene} />
    );
  }
}

type CanvasStageProps = Readonly<{
  displayList: DisplayList;
  camera: CameraTakeover | null;
  onHit(entityId: string): void;
}>;

function CanvasStage({
  displayList,
  camera,
  onHit,
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
    const cssWidth = Math.max(paintBounds.width, 1);
    const cssHeight = Math.max(paintBounds.height, 1);
    const width = Math.max(1, Math.ceil(cssWidth * ratio));
    const height = Math.max(1, Math.ceil(cssHeight * ratio));
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, width, height);
    context.save();
    context.scale(ratio, ratio);
    context.translate(-paintBounds.x, -paintBounds.y);
    if (camera !== null) {
      const centerX = paintBounds.x + paintBounds.width / 2;
      const centerY = paintBounds.y + paintBounds.height / 2;
      context.translate(centerX, centerY);
      context.scale(camera.temporary.zoom, camera.temporary.zoom);
      context.translate(
        -camera.temporary.x,
        -camera.temporary.y,
      );
    }
    renderDisplayList(context, displayList, { devicePixelRatio: 1 });
    context.restore();
  }, [camera, displayList]);

  function pointerToScene(clientX: number, clientY: number): {
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
        ((clientX - rect.left) / rect.width) * paintBounds.width + paintBounds.x,
      y:
        ((clientY - rect.top) / rect.height) * paintBounds.height +
        paintBounds.y,
    };
  }

  return (
    <canvas
      aria-hidden="true"
      className="aiperf-flow__canvas aiperf-flow__stage"
      data-backend="canvas"
      onPointerDown={(event) => {
        const point = pointerToScene(event.clientX, event.clientY);
        if (point === null) {
          return;
        }
        const hit = hitTest(displayList.hitRegions, point);
        if (hit !== undefined) {
          onHit(hit.semanticId);
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
  camera: CameraTakeover | null;
  inspectorEntity: SemanticEntityProjection | undefined;
  onCloseInspector(): void;
  subtitleState: SubtitleState;
  reducedMotion: boolean;
  onSubtitlesEnabledChange(enabled: boolean): void;
  onFocus(entityId: string): void;
  onActivate(entityId: string): void;
}>;

function CinematicStage({
  evaluated,
  projection,
  preferCanvas,
  focusedEntityId,
  selectedEntityId,
  camera,
  inspectorEntity,
  onCloseInspector,
  subtitleState,
  reducedMotion,
  onSubtitlesEnabledChange,
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
          camera={camera}
          displayList={evaluated.displayList}
          onHit={onActivate}
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
      {inspectorEntity === undefined ? null : (
        <aside
          aria-label="Node inspector"
          className="aiperf-flow__inspector"
          role="region"
          tabIndex={-1}
        >
          <strong>{inspectorEntity.label}</strong>
          {inspectorEntity.description === undefined ? null : (
            <p>{inspectorEntity.description}</p>
          )}
          <button onClick={onCloseInspector} type="button">
            Close inspector
          </button>
        </aside>
      )}
      <SubtitleOverlay
        onEnabledChange={onSubtitlesEnabledChange}
        reducedMotion={reducedMotion}
        state={subtitleState}
      />
    </section>
  );
}

type EvaluationResult =
  | { readonly ok: true; readonly scene: EvaluatedScene }
  | { readonly ok: false; readonly reason: string };

export type FlowAppProps = Readonly<{
  flow: FlowIr;
  registry?: CapabilityRegistry;
  clock?: Clock;
  narratorBackend?: NarratorBackend;
  reducedMotion?: boolean;
  /** Force SVG fallback even when Canvas 2D is available (tests). */
  forceSvgFallback?: boolean;
}>;

export function FlowApp({
  flow,
  registry: suppliedRegistry,
  clock: suppliedClock,
  narratorBackend: suppliedNarratorBackend,
  reducedMotion = false,
  forceSvgFallback = false,
}: FlowAppProps): ReactNode {
  const registry = useMemo(
    () => suppliedRegistry ?? createFoundationRegistry(),
    [suppliedRegistry],
  );
  const clock = useMemo(
    () => suppliedClock ?? new PerformanceClock(),
    [suppliedClock],
  );
  const narratorBackend = useMemo(
    () =>
      suppliedNarratorBackend ??
      createBrowserSpeechSynthesisBackend() ??
      unavailableNarratorBackend,
    [suppliedNarratorBackend],
  );
  const preferCanvas = useMemo(
    () => !forceSvgFallback && canvas2dAvailable(),
    [forceSvgFallback],
  );
  const scenes = Array.isArray(record(flow).scenes)
    ? (record(flow).scenes as readonly SceneIr[])
    : [];
  const [sceneIndex, setSceneIndex] = useState(0);
  const [timeMs, setTimeMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(true);
  const [exploration, setExploration] = useState<ExplorationSnapshot | null>(
    null,
  );
  const [cameraTakeover, setCameraTakeover] = useState<CameraTakeover | null>(
    null,
  );
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [focusedEntityId, setFocusedEntityId] = useState<string | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const scene = scenes[sceneIndex];
  const timeline = scene === undefined ? [] : record(scene).timeline;
  const narrativeCues = useMemo(
    () => (scene === undefined ? [] : sceneNarrativeCues(scene)),
    [scene],
  );
  const mutedRef = useRef(muted);
  mutedRef.current = muted;
  const narratorRef = useRef<NarratorController | null>(null);
  const focusCoordinatorRef = useRef<FocusCoordinator | null>(null);
  const narrator = useMemo(() => {
    const controller = new NarratorController(
      narrativeCues.map((cue) =>
        Object.freeze({
          id: cue.id,
          atMs: cue.atMs,
          text: cue.spokenText,
        }),
      ),
      narratorBackend,
    );
    if (mutedRef.current) {
      controller.setMuted(true);
    }
    narratorRef.current = controller;
    return controller;
  }, [narrativeCues, narratorBackend]);

  const player = useMemo(
    () =>
      new TimelinePlayer(
        (Array.isArray(timeline) ? timeline : []) as SceneIr["timeline"],
        clock,
        (snapshot) => {
          setTimeMs(snapshot.timeMs);
          narratorRef.current?.sync(snapshot.timeMs);
          if (snapshot.complete) {
            setPlaying(false);
          }
        },
      ),
    [timeline, clock],
  );

  useEffect(() => () => {
    player.pause();
    narratorRef.current?.stop();
  }, [player]);

  useEffect(() => () => {
    narrator.stop();
  }, [narrator]);

  useEffect(() => {
    setFocusedEntityId(null);
    setSelectedEntityId(null);
    setExploration(null);
    setCameraTakeover(null);
    setInspectorOpen(false);
    focusCoordinatorRef.current = null;
  }, [sceneIndex]);

  const properties = scene === undefined ? {} : record(scene);
  const transcript = scene === undefined ? "" : sceneTranscript(scene);
  const missing = useMemo(
    () => unavailableCapabilities(flow, registry),
    [flow, registry],
  );
  const validScene = Array.isArray(properties.roots);
  const exploring = exploration !== null;
  const evaluationTimeMs = exploring
    ? Math.trunc(exploration.authored.playbackTimeMs)
    : Math.trunc(timeMs);
  const narrativeTimeline = useMemo(
    () =>
      evaluateNarrativeTimeline(narrativeCues, evaluationTimeMs, {
        reducedMotion,
      }),
    [evaluationTimeMs, narrativeCues, reducedMotion],
  );
  const subtitleState = useMemo(
    (): SubtitleState => ({
      enabled: subtitlesEnabled,
      activeCue:
        narrativeTimeline.activeCue === null
          ? null
          : {
              id: narrativeTimeline.activeCue.id,
              text:
                narrativeTimeline.subtitleText ??
                narrativeTimeline.activeCue.spokenText,
            },
    }),
    [narrativeTimeline, subtitlesEnabled],
  );

  const evaluation = useMemo((): EvaluationResult | null => {
    if (scene === undefined || !validScene || missing.length > 0) {
      return null;
    }
    try {
      return {
        ok: true,
        scene: withDisplayList(evaluateScene(scene, evaluationTimeMs)),
      };
    } catch (error) {
      return {
        ok: false,
        reason:
          error instanceof Error
            ? error.message
            : "Scene evaluation failed.",
      };
    }
  }, [scene, evaluationTimeMs, validScene, missing]);

  const twinProjection = useMemo(() => {
    if (evaluation?.ok !== true) {
      return null;
    }
    return twinProjectionFromEvaluated(evaluation.scene, subtitleState);
  }, [evaluation, subtitleState]);

  useEffect(() => {
    if (twinProjection === null) {
      focusCoordinatorRef.current = null;
      return;
    }
    focusCoordinatorRef.current = createFocusCoordinator(twinProjection);
  }, [twinProjection]);

  if (scene === undefined) {
    return (
      <main className="aiperf-flow">
        <h1>{text(record(flow).title, "AIPerf Flow")}</h1>
        <p>No scenes are available.</p>
      </main>
    );
  }

  const displayTimeMs = exploring
    ? exploration.authored.playbackTimeMs
    : timeMs;
  const highlightText = subtitleState.enabled
    ? (subtitleState.activeCue?.text ?? "")
    : "";
  const inspectorEntity = inspectorOpen
    ? entityFromProjection(twinProjection, selectedEntityId)
    : undefined;

  function applyFocusState(next: {
    focusedEntityId: string | null;
    selectedEntityId: string | null;
  }): void {
    setFocusedEntityId(next.focusedEntityId);
    setSelectedEntityId(next.selectedEntityId);
  }

  function navigate(nextIndex: number): void {
    narrator.stop();
    player.reset();
    setPlaying(false);
    setTimeMs(0);
    setExploration(null);
    setCameraTakeover(null);
    setInspectorOpen(false);
    setSceneIndex(nextIndex);
  }

  function togglePlayback(): void {
    if (exploring) {
      return;
    }
    if (playing) {
      const snapshot = player.pause();
      narrator.pause(snapshot.timeMs);
      setTimeMs(snapshot.timeMs);
      setPlaying(false);
      return;
    }
    const snapshot = player.play();
    narrator.play(snapshot.timeMs);
    setTimeMs(snapshot.timeMs);
    setPlaying(true);
  }

  function restart(): void {
    narrator.stop();
    player.reset();
    setTimeMs(0);
    setExploration(null);
    setCameraTakeover(null);
    setInspectorOpen(false);
    setPlaying(false);
  }

  function toggleMute(): void {
    const nextMuted = !muted;
    narrator.setMuted(nextMuted);
    setMuted(nextMuted);
  }

  function startExploration(): void {
    if (exploring) {
      return;
    }
    const wasPlaying = playing;
    const snapshot = player.pause();
    narrator.pause(snapshot.timeMs);
    setTimeMs(snapshot.timeMs);
    setPlaying(false);
    setExploration(
      beginExploration({
        ...createInitialSceneState(text(properties.id)),
        selectedNodeId: selectedEntityId,
        playbackTimeMs: snapshot.timeMs,
        playbackStatus: wasPlaying ? "playing" : "paused",
      }),
    );
    const cameraTrack = Array.isArray(scene?.camera) ? scene.camera : [];
    setCameraTakeover(beginCameraTakeover(cameraTrack, snapshot.timeMs));
  }

  function resumeFromExploration(): void {
    if (exploration === null) {
      return;
    }
    const restored = resumeLesson(exploration);
    if (cameraTakeover !== null) {
      resumeAuthoredCamera(cameraTakeover, { reducedMotion });
    }
    setCameraTakeover(null);
    setInspectorOpen(false);
    const seeked = player.seek(restored.playbackTimeMs);
    setTimeMs(seeked.timeMs);
    setExploration(null);
    const coordinator = focusCoordinatorRef.current;
    if (restored.selectedNodeId === null) {
      applyFocusState(coordinator?.clear() ?? {
        focusedEntityId: null,
        selectedEntityId: null,
      });
    } else {
      applyFocusState(
        coordinator?.restore(restored.selectedNodeId) ?? {
          focusedEntityId: restored.selectedNodeId,
          selectedEntityId: restored.selectedNodeId,
        },
      );
    }
    if (restored.playbackStatus === "playing") {
      narrator.resume(seeked.timeMs);
      const played = player.play();
      setTimeMs(played.timeMs);
      setPlaying(true);
    } else {
      narrator.seek(seeked.timeMs);
      setPlaying(false);
    }
  }

  function focusEntity(entityId: string): void {
    const coordinator = focusCoordinatorRef.current;
    if (coordinator === null) {
      setFocusedEntityId(entityId);
      return;
    }
    const next = coordinator.selectFromVisual(entityId);
    setFocusedEntityId(next.focusedEntityId);
  }

  function activateEntity(entityId: string): void {
    const coordinator = focusCoordinatorRef.current;
    const next =
      coordinator?.activateFromSemantic(entityId) ??
      ({
        focusedEntityId: entityId,
        selectedEntityId: entityId,
        visualSelectedEntityId: entityId,
      } as const);
    applyFocusState(next);
    if (exploration !== null) {
      setExploration(
        updateExploration(exploration, {
          ...exploration.exploration,
          selectedNodeId: entityId,
        }),
      );
    }
  }

  function openInspector(): void {
    if (selectedEntityId === null) {
      return;
    }
    setInspectorOpen(true);
    if (exploration !== null) {
      setExploration(
        updateExploration(exploration, {
          ...exploration.exploration,
          selectedNodeId: selectedEntityId,
          inspector: { open: true, nodeId: selectedEntityId },
        }),
      );
    }
  }

  function closeInspector(): void {
    setInspectorOpen(false);
    if (exploration !== null) {
      setExploration(
        updateExploration(exploration, {
          ...exploration.exploration,
          inspector: { open: false, nodeId: null },
        }),
      );
    }
  }

  function panExplore(): void {
    if (cameraTakeover === null) {
      return;
    }
    setCameraTakeover(panCameraTakeover(cameraTakeover, { x: 24, y: 0 }));
  }

  function zoomExplore(): void {
    if (cameraTakeover === null) {
      return;
    }
    setCameraTakeover(
      zoomCameraTakeover(cameraTakeover, cameraTakeover.temporary.zoom * 1.25),
    );
  }

  function fitExplore(): void {
    if (cameraTakeover === null || evaluation?.ok !== true) {
      return;
    }
    setCameraTakeover(
      fitCameraTakeover(
        cameraTakeover,
        evaluation.scene.displayList.paintBounds,
        {
          width: Math.max(evaluation.scene.displayList.paintBounds.width, 1),
          height: Math.max(evaluation.scene.displayList.paintBounds.height, 1),
        },
        16,
      ),
    );
  }

  let stage: ReactNode;
  if (!validScene || missing.length > 0) {
    stage = (
      <SceneFailure
        reason={
          missing.length === 0
            ? "The scene chunk is invalid."
            : `Missing capabilities: ${missing.join(", ")}`
        }
        scene={scene}
      />
    );
  } else if (evaluation === null) {
    stage = <SceneFailure scene={scene} />;
  } else if (!evaluation.ok) {
    stage = <SceneFailure reason={evaluation.reason} scene={scene} />;
  } else if (twinProjection === null) {
    stage = <SceneFailure scene={scene} />;
  } else {
    stage = (
      <CinematicStage
        camera={cameraTakeover}
        evaluated={evaluation.scene}
        focusedEntityId={focusedEntityId}
        inspectorEntity={inspectorEntity}
        onActivate={activateEntity}
        onCloseInspector={closeInspector}
        onFocus={focusEntity}
        onSubtitlesEnabledChange={setSubtitlesEnabled}
        preferCanvas={preferCanvas}
        projection={twinProjection}
        reducedMotion={reducedMotion}
        selectedEntityId={selectedEntityId}
        subtitleState={subtitleState}
      />
    );
  }

  return (
    <main className="aiperf-flow">
      <a className="aiperf-flow__skip-link" href="#flow-transcript">
        Skip to transcript
      </a>
      <div className="aiperf-flow__shell">
        <header className="aiperf-flow__header aiperf-flow__chrome">
          <div>
            <p className="aiperf-flow__eyebrow">
              Scene {sceneIndex + 1} of {scenes.length}
            </p>
            <h1>{text(properties.title, "Untitled scene")}</h1>
          </div>
          <nav aria-label="Scene navigation" className="aiperf-flow__navigation">
            <button
              disabled={sceneIndex === 0}
              onClick={() => navigate(sceneIndex - 1)}
              type="button"
            >
              Previous scene
            </button>
            <button
              disabled={sceneIndex >= scenes.length - 1}
              onClick={() => navigate(sceneIndex + 1)}
              type="button"
            >
              Next scene
            </button>
          </nav>
        </header>

        <section
          aria-label="Playback controls"
          className="aiperf-flow__controls aiperf-flow__chrome"
        >
          <button disabled={exploring} onClick={togglePlayback} type="button">
            {playing ? "Pause" : "Play"}
          </button>
          <button onClick={restart} type="button">
            Restart
          </button>
          <button aria-pressed={muted} onClick={toggleMute} type="button">
            {muted ? "Unmute narration" : "Mute narration"}
          </button>
          {exploring ? (
            <>
              <button onClick={resumeFromExploration} type="button">
                Resume lesson
              </button>
              <button
                disabled={selectedEntityId === null}
                onClick={openInspector}
                type="button"
              >
                Inspect
              </button>
              <button onClick={panExplore} type="button">
                Pan
              </button>
              <button onClick={zoomExplore} type="button">
                Zoom in
              </button>
              <button onClick={fitExplore} type="button">
                Fit
              </button>
            </>
          ) : (
            <button onClick={startExploration} type="button">
              Explore
            </button>
          )}
          <output aria-label="Playback time" aria-live="off" role="status">
            {Math.round(displayTimeMs)} ms
          </output>
        </section>

        <div className="aiperf-flow__stage-region">
          <SceneErrorBoundary key={text(properties.id)} scene={scene}>
            {stage}
          </SceneErrorBoundary>
        </div>

        <p
          aria-atomic="true"
          aria-live="polite"
          className="aiperf-flow__narration-highlight"
        >
          {highlightText}
        </p>
        <section
          aria-label="Narration transcript"
          className="aiperf-flow__transcript"
          id="flow-transcript"
          tabIndex={-1}
        >
          <h2>Transcript</h2>
          <p>{transcript}</p>
        </section>
      </div>
    </main>
  );
}
