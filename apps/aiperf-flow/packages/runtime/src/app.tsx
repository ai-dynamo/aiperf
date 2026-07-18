// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import {
  Component,
  type ErrorInfo,
  type ReactNode,
  type RefObject,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import { renderDisplayList } from "./backends/canvas/canvas-renderer.js";
import { hitTest } from "./backends/canvas/hit-test.js";
import { CanvasTextAtlas } from "./backends/canvas/text-atlas.js";
import { SvgFallback } from "./backends/svg/svg-fallback.js";
import {
  beginCameraTakeover,
  type CameraTakeover,
  fitCameraTakeover,
  panCameraTakeover,
  resumeAuthoredCamera,
  zoomCameraTakeover,
} from "./camera-policy.js";
import {
  activeCausalBeat,
  projectCausalBeats,
  type CausalBeat,
} from "./causal-replay.js";
import type { FlowCommand } from "./commands.js";
import type { DisplayList } from "./display-list.js";
import { evaluateFrame, type EvaluatedFrame } from "./evaluate/frame.js";
import { qualityPolicyProfile } from "./evaluate/quality-policy.js";
import type {
  EvaluatedScene,
  SemanticProjection,
} from "./evaluate/types.js";
import {
  beginExploration,
  type ExplorationSnapshot,
  resumeLesson,
  updateExploration,
} from "./exploration.js";
import {
  createBrowserFullscreenAdapter,
  type FullscreenAdapter,
  resolveFullscreenState,
  toggleFullscreenMode,
} from "./fullscreen.js";
import { hudVisibilityFor } from "./hud-policy.js";
import { CausalPath } from "./immersive/causal-path.js";
import { CommandConstellation } from "./immersive/command-constellation.js";
import { ContextLens } from "./immersive/context-lens.js";
import { ImmersiveControls } from "./immersive/immersive-controls.js";
import {
  createImmersiveState,
  immersiveReducer,
  type ImmersiveState,
} from "./immersive-state.js";
import {
  parseImmersiveUrl,
  serializeImmersiveUrl,
} from "./immersive-url.js";
import {
  AudioConsentModal,
  type AudioConsentChoice,
} from "./narrative/audio-consent-modal.js";
import {
  createBrowserSpeechSynthesisBackend,
  NarratorController,
  type NarratorBackend,
} from "./narrative/narrator.js";
import {
  createKokoroNarratorBackend,
  type KokoroNarratorBackend,
} from "./narrative/kokoro-narrator.js";
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

const HUD_INACTIVITY_MS = 3_000;

const unavailableNarratorBackend: NarratorBackend = Object.freeze({
  available: false,
  voices: () => Object.freeze([]),
  speak: () => undefined,
  pause: () => undefined,
  resume: () => undefined,
  cancel: () => undefined,
});

function unlockSpeechFromGesture(backend: NarratorBackend): void {
  const kokoro = backend as Partial<KokoroNarratorBackend>;
  if (typeof kokoro.prewarm === "function") {
    void Promise.resolve(kokoro.prewarm()).catch(() => undefined);
  }
  if (typeof kokoro.activate === "function") {
    void Promise.resolve(kokoro.activate()).catch(() => undefined);
  }
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return;
  }
  try {
    window.speechSynthesis.getVoices();
    const prime = new SpeechSynthesisUtterance(" ");
    prime.volume = 0;
    window.speechSynthesis.speak(prime);
    window.speechSynthesis.cancel();
  } catch {
    // Browser speech unlock is best-effort; Kokoro activation is primary.
  }
}

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

function sceneIdOf(scene: SceneIr | undefined): string {
  return scene === undefined ? "" : text(record(scene).id);
}

function freezeCommands(commands: FlowCommand[]): readonly FlowCommand[] {
  return Object.freeze(commands.map((command) => Object.freeze(command)));
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
  const textAtlasRef = useRef<Readonly<{
    context: CanvasRenderingContext2D;
    atlas: CanvasTextAtlas;
  }> | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const context = canvas.getContext("2d");
    if (context === null) {
      return;
    }
    if (textAtlasRef.current?.context !== context) {
      textAtlasRef.current = {
        context,
        atlas: new CanvasTextAtlas(context),
      };
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
      context.translate(-camera.temporary.x, -camera.temporary.y);
    }
    renderDisplayList(context, displayList, {
      devicePixelRatio: 1,
      textAtlas: textAtlasRef.current.atlas,
    });
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

type CausalFieldStageProps = Readonly<{
  evaluated: EvaluatedScene;
  projection: SemanticProjection;
  preferCanvas: boolean;
  focusedEntityId: string | null;
  selectedEntityId: string | null;
  focusWorldEntityId: string | null;
  camera: CameraTakeover | null;
  contextLensOpen: boolean;
  twinCompact: boolean;
  twinRef: RefObject<HTMLElement | null>;
  subtitleState: SubtitleState;
  reducedMotion: boolean;
  onSubtitlesEnabledChange(enabled: boolean): void;
  onFocus(entityId: string): void;
  onActivate(entityId: string): void;
  onCloseContext(): void;
  onFocusWorld(entityId: string): void;
  onOpenTwin(entityId: string): void;
}>;

function CausalFieldStage({
  evaluated,
  projection,
  preferCanvas,
  focusedEntityId,
  selectedEntityId,
  focusWorldEntityId,
  camera,
  contextLensOpen,
  twinCompact,
  twinRef,
  subtitleState,
  reducedMotion,
  onSubtitlesEnabledChange,
  onFocus,
  onActivate,
  onCloseContext,
  onFocusWorld,
  onOpenTwin,
}: CausalFieldStageProps): ReactNode {
  const lensEntityId =
    contextLensOpen && selectedEntityId !== null ? selectedEntityId : null;

  return (
    <section
      aria-label="Scene field"
      className="aiperf-flow__scene"
      data-backend={preferCanvas ? "canvas" : "svg"}
      data-focus-world={focusWorldEntityId === null ? undefined : "true"}
      data-focus-world-entity={focusWorldEntityId ?? undefined}
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
      <div ref={twinRef as RefObject<HTMLDivElement>}>
        <SemanticTwin
          compact={twinCompact}
          focusedEntityId={focusedEntityId}
          onActivate={onActivate}
          onFocus={onFocus}
          projection={projection}
          selectedEntityId={selectedEntityId}
        />
      </div>
      {lensEntityId === null ? null : (
        <ContextLens
          entityId={lensEntityId}
          onClose={onCloseContext}
          onFocusWorld={onFocusWorld}
          onOpenTwin={onOpenTwin}
          projection={projection}
        />
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
  | { readonly ok: true; readonly frame: EvaluatedFrame }
  | { readonly ok: false; readonly reason: string };

export type FlowAppProps = Readonly<{
  flow: FlowIr;
  registry?: CapabilityRegistry;
  clock?: Clock;
  narratorBackend?: NarratorBackend;
  reducedMotion?: boolean;
  /** Force SVG fallback even when Canvas 2D is available (tests). */
  forceSvgFallback?: boolean;
  /**
   * When true (default), block playback behind an audio consent dialog so
   * "Play with audio" can unlock Web Audio from a user gesture.
   */
  requireAudioConsent?: boolean;
  /** Injectable Fullscreen API boundary for tests and hosts. */
  fullscreenAdapter?: FullscreenAdapter;
}>;

export function FlowApp({
  flow,
  registry: suppliedRegistry,
  clock: suppliedClock,
  narratorBackend: suppliedNarratorBackend,
  reducedMotion = false,
  forceSvgFallback = false,
  requireAudioConsent = suppliedNarratorBackend === undefined && !forceSvgFallback,
  fullscreenAdapter: suppliedFullscreenAdapter,
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
      createKokoroNarratorBackend({
        fallback:
          createBrowserSpeechSynthesisBackend() ??
          unavailableNarratorBackend,
      }),
    [suppliedNarratorBackend],
  );
  const fullscreenAdapter = useMemo(
    () => suppliedFullscreenAdapter ?? createBrowserFullscreenAdapter(),
    [suppliedFullscreenAdapter],
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
  const [audioConsent, setAudioConsent] = useState<AudioConsentChoice | null>(
    () => (requireAudioConsent ? null : "with-audio"),
  );
  const [muted, setMuted] = useState(audioConsent === "without-audio");
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(true);
  const [exploration, setExploration] = useState<ExplorationSnapshot | null>(
    null,
  );
  const [cameraTakeover, setCameraTakeover] = useState<CameraTakeover | null>(
    null,
  );
  const [focusedEntityId, setFocusedEntityId] = useState<string | null>(null);
  const [immersive, dispatchImmersive] = useReducer(
    immersiveReducer,
    undefined,
    createImmersiveState,
  );
  const [twinCompact, setTwinCompact] = useState(true);
  const [hudInactive, setHudInactive] = useState(false);
  const [focusedWithinHud, setFocusedWithinHud] = useState(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState("");
  const [urlReady, setUrlReady] = useState(false);

  const rootRef = useRef<HTMLElement | null>(null);
  const twinRegionRef = useRef<HTMLElement | null>(null);
  const previousDisplayListRef = useRef<DisplayList | undefined>(undefined);
  const urlAppliedRef = useRef(false);

  const scene = scenes[sceneIndex];
  const sceneId = sceneIdOf(scene);
  const timeline = scene === undefined ? [] : record(scene).timeline;
  const narrativeCues = useMemo(
    () => (scene === undefined ? [] : sceneNarrativeCues(scene)),
    [scene],
  );
  const causalBeats = useMemo((): readonly CausalBeat[] => {
    if (scene === undefined) {
      return Object.freeze([]);
    }
    try {
      return projectCausalBeats(scene);
    } catch {
      return Object.freeze([]);
    }
  }, [scene]);

  const mutedRef = useRef(muted);
  mutedRef.current = muted;
  const narratorRef = useRef<NarratorController | null>(null);
  const focusCoordinatorRef = useRef<FocusCoordinator | null>(null);
  const immersiveRef = useRef(immersive);
  immersiveRef.current = immersive;

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
    setExploration(null);
    setCameraTakeover(null);
    focusCoordinatorRef.current = null;
    previousDisplayListRef.current = undefined;
    dispatchImmersive({ type: "select", entityId: null });
    dispatchImmersive({ type: "close-context" });
    if (immersiveRef.current.focusWorldEntityId !== null) {
      dispatchImmersive({ type: "leave-focus-world" });
    }
    if (immersiveRef.current.commandOpen) {
      dispatchImmersive({ type: "close-command" });
    }
  }, [sceneIndex]);

  useEffect(() => {
    if (
      audioConsent === null ||
      exploration !== null ||
      reducedMotion ||
      !playing
    ) {
      return;
    }
    const snapshot = player.play();
    setTimeMs(snapshot.timeMs);
    if (!muted) {
      narrator.play(snapshot.timeMs);
    }
  }, [
    audioConsent,
    exploration,
    muted,
    narrator,
    player,
    playing,
    reducedMotion,
  ]);

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
  const qualityProfile = useMemo(
    () =>
      qualityPolicyProfile("reference", {
        motion: reducedMotion ? "reduced" : "full",
      }),
    [reducedMotion],
  );
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
        frame: evaluateFrame(scene, evaluationTimeMs, {
          quality: qualityProfile,
          previousDisplayList: previousDisplayListRef.current,
        }),
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
  }, [
    scene,
    sceneId,
    evaluationTimeMs,
    validScene,
    missing,
    qualityProfile,
  ]);

  useEffect(() => {
    if (evaluation?.ok === true) {
      previousDisplayListRef.current = evaluation.frame.displayList;
    }
  }, [evaluation]);

  const twinProjection = useMemo(() => {
    if (evaluation?.ok !== true) {
      return null;
    }
    return twinProjectionFromEvaluated(evaluation.frame.scene, subtitleState);
  }, [evaluation, subtitleState]);

  useEffect(() => {
    if (twinProjection === null) {
      focusCoordinatorRef.current = null;
      return;
    }
    focusCoordinatorRef.current = createFocusCoordinator(twinProjection);
  }, [twinProjection]);

  const policyHud = hudVisibilityFor({
    playing,
    exploring,
    commandOpen: immersive.commandOpen,
    focusedWithinHud,
    inactive: hudInactive,
  });

  useEffect(() => {
    if (immersive.hud !== policyHud) {
      dispatchImmersive({ type: "set-hud", visibility: policyHud });
    }
  }, [immersive.hud, policyHud]);

  useEffect(() => {
    if (!playing || exploring || immersive.commandOpen) {
      setHudInactive(false);
      return;
    }
    let timer = window.setTimeout(() => {
      setHudInactive(true);
    }, HUD_INACTIVITY_MS);
    const onActivity = (): void => {
      setHudInactive(false);
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        setHudInactive(true);
      }, HUD_INACTIVITY_MS);
    };
    window.addEventListener("pointermove", onActivity);
    window.addEventListener("pointerdown", onActivity);
    window.addEventListener("keydown", onActivity);
    window.addEventListener("touchstart", onActivity);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("pointermove", onActivity);
      window.removeEventListener("pointerdown", onActivity);
      window.removeEventListener("keydown", onActivity);
      window.removeEventListener("touchstart", onActivity);
    };
  }, [playing, exploring, immersive.commandOpen]);

  useEffect(() => {
    const onFullscreenChange = (): void => {
      const next = resolveFullscreenState(
        fullscreenAdapter,
        immersiveRef.current.fullscreen,
      );
      if (next !== immersiveRef.current.fullscreen) {
        dispatchImmersive({ type: "set-fullscreen", state: next });
      }
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", onFullscreenChange);
    };
  }, [fullscreenAdapter]);

  // Restore shareable scene/beat/entity selections from the URL once.
  useEffect(() => {
    if (urlAppliedRef.current || typeof window === "undefined") {
      return;
    }
    urlAppliedRef.current = true;
    const parsed = parseImmersiveUrl(window.location.search);
    let nextIndex = 0;
    if (parsed.sceneId !== null) {
      const matched = scenes.findIndex(
        (candidate) => sceneIdOf(candidate) === parsed.sceneId,
      );
      if (matched >= 0) {
        nextIndex = matched;
      }
    }
    if (nextIndex !== sceneIndex) {
      setSceneIndex(nextIndex);
    }
    const targetScene = scenes[nextIndex];
    if (targetScene !== undefined && parsed.beatId !== null) {
      try {
        const beats = projectCausalBeats(targetScene);
        const beat = beats.find((entry) => entry.id === parsed.beatId);
        if (beat !== undefined) {
          const seeked = player.seek(beat.timeMs);
          setTimeMs(seeked.timeMs);
          narrator.seek(seeked.timeMs);
        }
      } catch {
        // Invalid beat projection fails closed to the current authored time.
      }
    }
    if (parsed.entityId !== null) {
      dispatchImmersive({ type: "select", entityId: parsed.entityId });
      setFocusedEntityId(parsed.entityId);
    }
    setUrlReady(true);
  }, [narrator, player, sceneIndex, scenes]);

  const activeBeat = activeCausalBeat(causalBeats, evaluationTimeMs);

  useEffect(() => {
    if (!urlReady || typeof window === "undefined") {
      return;
    }
    const nextSearch = serializeImmersiveUrl({
      sceneId: sceneId === "" ? null : sceneId,
      beatId: activeBeat?.id ?? null,
      entityId: immersive.selectedEntityId,
    });
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    const desired = `${window.location.pathname}${nextSearch}${window.location.hash}`;
    if (current !== desired) {
      window.history.replaceState(null, "", desired);
    }
  }, [
    activeBeat?.id,
    immersive.selectedEntityId,
    sceneId,
    urlReady,
  ]);

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

  function applyFocusState(next: {
    focusedEntityId: string | null;
    selectedEntityId: string | null;
  }): void {
    setFocusedEntityId(next.focusedEntityId);
    dispatchImmersive({ type: "select", entityId: next.selectedEntityId });
  }

  function navigate(nextIndex: number): void {
    narrator.stop();
    player.reset();
    setTimeMs(0);
    setExploration(null);
    setCameraTakeover(null);
    setSceneIndex(nextIndex);
    if (audioConsent !== null && !reducedMotion) {
      setPlaying(true);
    } else {
      setPlaying(false);
    }
  }

  function chooseAudioConsent(choice: AudioConsentChoice): void {
    setAudioConsent(choice);
    if (choice === "with-audio") {
      setMuted(false);
      narrator.setMuted(false);
      unlockSpeechFromGesture(narratorBackend);
    } else {
      setMuted(true);
      narrator.setMuted(true);
    }
    if (!reducedMotion) {
      const snapshot = player.play();
      setTimeMs(snapshot.timeMs);
      if (choice === "with-audio") {
        narrator.seek(snapshot.timeMs);
        narrator.play(snapshot.timeMs);
      }
      setPlaying(true);
    }
  }

  function togglePlayback(): void {
    if (exploring || audioConsent === null) {
      return;
    }
    if (playing) {
      const snapshot = player.pause();
      narrator.pause(snapshot.timeMs);
      setTimeMs(snapshot.timeMs);
      setPlaying(false);
      return;
    }
    if (!muted) {
      void unlockSpeechFromGesture(narratorBackend);
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
    const immersiveSnapshot: ImmersiveState = immersive;
    setExploration(
      beginExploration({
        ...createInitialSceneState(sceneId),
        selectedNodeId: immersive.selectedEntityId,
        playbackTimeMs: snapshot.timeMs,
        playbackStatus: wasPlaying ? "playing" : "paused",
        immersive: immersiveSnapshot,
      }),
    );
    const cameraTrack = Array.isArray(scene?.camera) ? scene.camera : [];
    setCameraTakeover(beginCameraTakeover(cameraTrack, snapshot.timeMs));
  }

  function restoreImmersiveSnapshot(target: ImmersiveState): void {
    dispatchImmersive({ type: "replace", state: target });
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
    if (restored.immersive !== undefined) {
      restoreImmersiveSnapshot(restored.immersive);
    } else {
      dispatchImmersive({ type: "close-context" });
      if (immersive.focusWorldEntityId !== null) {
        dispatchImmersive({ type: "leave-focus-world" });
      }
    }
    const seeked = player.seek(restored.playbackTimeMs);
    setTimeMs(seeked.timeMs);
    setExploration(null);
    const coordinator = focusCoordinatorRef.current;
    if (restored.selectedNodeId === null) {
      applyFocusState(
        coordinator?.clear() ?? {
          focusedEntityId: null,
          selectedEntityId: null,
        },
      );
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

  function toggleExploreResume(): void {
    if (exploring) {
      resumeFromExploration();
      return;
    }
    startExploration();
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
    setFocusedEntityId(next.focusedEntityId);
    dispatchImmersive({ type: "open-context", entityId });
    if (exploration !== null) {
      setExploration(
        updateExploration(exploration, {
          ...exploration.exploration,
          selectedNodeId: entityId,
          immersive: {
            ...immersive,
            selectedEntityId: entityId,
            contextLensOpen: true,
          },
        }),
      );
    }
  }

  function closeContextLens(): void {
    dispatchImmersive({ type: "close-context" });
    if (exploration !== null) {
      setExploration(
        updateExploration(exploration, {
          ...exploration.exploration,
          immersive: {
            ...immersive,
            contextLensOpen: false,
          },
        }),
      );
    }
  }

  function enterFocusWorld(entityId: string): void {
    dispatchImmersive({ type: "enter-focus-world", entityId });
    setFocusedEntityId(entityId);
  }

  function leaveFocusWorld(): void {
    if (immersive.focusWorldEntityId === null) {
      return;
    }
    dispatchImmersive({ type: "leave-focus-world" });
  }

  function openTwin(entityId: string): void {
    setTwinCompact(false);
    applyFocusState({
      focusedEntityId: entityId,
      selectedEntityId: entityId,
    });
    const twinRoot = twinRegionRef.current;
    if (twinRoot === null) {
      return;
    }
    for (const node of twinRoot.querySelectorAll<HTMLElement>("[data-entity-id]")) {
      if (node.getAttribute("data-entity-id") === entityId) {
        node.focus();
        break;
      }
    }
  }

  function seekBeat(beatTimeMs: number, _beatId: string): void {
    const seeked = player.seek(beatTimeMs);
    setTimeMs(seeked.timeMs);
    if (playing && !exploring) {
      narrator.seek(seeked.timeMs);
      narrator.play(seeked.timeMs);
    } else {
      narrator.seek(seeked.timeMs);
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
        evaluation.frame.displayList.paintBounds,
        {
          width: Math.max(evaluation.frame.displayList.paintBounds.width, 1),
          height: Math.max(evaluation.frame.displayList.paintBounds.height, 1),
        },
        16,
      ),
    );
  }

  async function toggleFullscreen(): Promise<void> {
    const element = rootRef.current;
    if (element === null) {
      return;
    }
    const result = await toggleFullscreenMode(
      fullscreenAdapter,
      element,
      immersive.fullscreen,
    );
    dispatchImmersive({ type: "set-fullscreen", state: result.state });
    if (result.announcement !== null) {
      setLiveAnnouncement(result.announcement);
    }
  }

  const commands = ((): readonly FlowCommand[] => {
    const catalog: FlowCommand[] = [];

    scenes.forEach((candidate, index) => {
      const id = sceneIdOf(candidate);
      const title = text(record(candidate).title, id || `Scene ${index + 1}`);
      catalog.push({
        id: `scene:${id || String(index)}`,
        label: title,
        category: "scene",
        keywords: Object.freeze(["scene", id, title]),
        execute: () => {
          navigate(index);
          dispatchImmersive({ type: "close-command" });
        },
      });
    });

    for (const beat of causalBeats) {
      catalog.push({
        id: `beat:${beat.id}`,
        label: beat.label,
        category: "beat",
        keywords: Object.freeze(["beat", beat.id, beat.label]),
        execute: () => {
          seekBeat(beat.timeMs, beat.id);
          dispatchImmersive({ type: "close-command" });
        },
      });
    }

    if (twinProjection !== null) {
      for (const entity of twinProjection.entities) {
        catalog.push({
          id: `entity:${entity.id}`,
          label: entity.label,
          category: "entity",
          keywords: Object.freeze([
            "entity",
            entity.id,
            entity.label,
            entity.role ?? "",
            entity.kind ?? "",
          ]),
          execute: () => {
            activateEntity(entity.id);
            dispatchImmersive({ type: "close-command" });
          },
        });
        for (const evidenceId of entity.evidenceIds ?? []) {
          catalog.push({
            id: `evidence:${evidenceId}`,
            label: `Evidence ${evidenceId}`,
            category: "evidence",
            keywords: Object.freeze(["evidence", evidenceId, entity.label]),
            execute: () => {
              activateEntity(entity.id);
              dispatchImmersive({ type: "close-command" });
            },
          });
        }
      }
    }

    catalog.push(
      {
        id: "action:play-pause",
        label: playing ? "Pause" : "Play",
        category: "action",
        keywords: Object.freeze(["play", "pause", "playback"]),
        shortcut: "Space",
        disabledReason:
          exploring
            ? "Pause exploration before changing playback."
            : audioConsent === null
              ? "Choose an audio preference first."
              : undefined,
        execute: () => {
          togglePlayback();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:restart",
        label: "Restart",
        category: "action",
        keywords: Object.freeze(["restart", "reset"]),
        execute: () => {
          restart();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:explore-resume",
        label: exploring ? "Resume lesson" : "Explore",
        category: "action",
        keywords: Object.freeze(["explore", "resume", "lesson"]),
        execute: () => {
          toggleExploreResume();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:mute",
        label: muted ? "Unmute narration" : "Mute narration",
        category: "accessibility",
        keywords: Object.freeze(["mute", "unmute", "narration", "audio"]),
        execute: () => {
          toggleMute();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:captions",
        label: subtitlesEnabled ? "Hide captions" : "Show captions",
        category: "accessibility",
        keywords: Object.freeze(["captions", "subtitles"]),
        execute: () => {
          setSubtitlesEnabled((enabled) => !enabled);
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:twin",
        label: twinCompact ? "Expand semantic twin" : "Compact semantic twin",
        category: "accessibility",
        keywords: Object.freeze(["twin", "semantic", "outline"]),
        execute: () => {
          setTwinCompact((compact) => !compact);
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:fullscreen",
        label:
          immersive.fullscreen === "windowed"
            ? "Enter fullscreen"
            : "Exit fullscreen",
        category: "action",
        keywords: Object.freeze(["fullscreen", "immersive"]),
        execute: () => {
          void toggleFullscreen();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:pan",
        label: "Pan",
        category: "action",
        keywords: Object.freeze(["pan", "camera", "explore"]),
        disabledReason: exploring ? undefined : "Start exploration to pan.",
        execute: () => {
          panExplore();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:zoom",
        label: "Zoom in",
        category: "action",
        keywords: Object.freeze(["zoom", "camera", "explore"]),
        disabledReason: exploring ? undefined : "Start exploration to zoom.",
        execute: () => {
          zoomExplore();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:fit",
        label: "Fit",
        category: "action",
        keywords: Object.freeze(["fit", "camera", "explore"]),
        disabledReason: exploring ? undefined : "Start exploration to fit.",
        execute: () => {
          fitExplore();
          dispatchImmersive({ type: "close-command" });
        },
      },
      {
        id: "action:leave-focus-world",
        label: "Leave Focus World",
        category: "action",
        keywords: Object.freeze(["focus", "world", "leave"]),
        disabledReason:
          immersive.focusWorldEntityId === null
            ? "Focus World is not active."
            : undefined,
        execute: () => {
          leaveFocusWorld();
          dispatchImmersive({ type: "close-command" });
        },
      },
    );

    return freezeCommands(catalog);
  })();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const meta = event.metaKey || event.ctrlKey;
      if (meta && event.key.toLowerCase() === "k") {
        event.preventDefault();
        dispatchImmersive({ type: "open-command" });
        return;
      }
      if (event.key === "Escape") {
        if (immersiveRef.current.commandOpen) {
          dispatchImmersive({ type: "close-command" });
          return;
        }
        if (immersiveRef.current.contextLensOpen) {
          dispatchImmersive({ type: "close-context" });
          return;
        }
        if (immersiveRef.current.focusWorldEntityId !== null) {
          dispatchImmersive({ type: "leave-focus-world" });
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

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
      <CausalFieldStage
        camera={cameraTakeover}
        contextLensOpen={immersive.contextLensOpen}
        evaluated={evaluation.frame.scene}
        focusedEntityId={focusedEntityId}
        focusWorldEntityId={immersive.focusWorldEntityId}
        onActivate={activateEntity}
        onCloseContext={closeContextLens}
        onFocus={focusEntity}
        onFocusWorld={enterFocusWorld}
        onOpenTwin={openTwin}
        onSubtitlesEnabledChange={setSubtitlesEnabled}
        preferCanvas={preferCanvas}
        projection={twinProjection}
        reducedMotion={reducedMotion}
        selectedEntityId={immersive.selectedEntityId}
        subtitleState={subtitleState}
        twinCompact={twinCompact}
        twinRef={twinRegionRef}
      />
    );
  }

  return (
    <main
      className="aiperf-flow"
      data-focus-world={
        immersive.focusWorldEntityId === null ? undefined : "true"
      }
      data-fullscreen={immersive.fullscreen}
      data-hud={policyHud}
      ref={rootRef}
    >
      <AudioConsentModal
        onChoose={chooseAudioConsent}
        open={audioConsent === null}
      />
      <a className="aiperf-flow__skip-link" href="#flow-transcript">
        Skip to transcript
      </a>
      <div className="aiperf-flow__shell aiperf-flow__causal-field">
        <header className="aiperf-flow__header aiperf-flow__chrome">
          <div>
            <p className="aiperf-flow__eyebrow">
              Scene {sceneIndex + 1} of {scenes.length}
              {activeBeat === null ? null : ` · ${activeBeat.label}`}
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
          <output aria-label="Playback time" aria-live="off" role="status">
            {Math.round(displayTimeMs)} ms
          </output>
        </header>

        <div className="aiperf-flow__stage-region">
          <SceneErrorBoundary key={sceneId} scene={scene}>
            {stage}
          </SceneErrorBoundary>
        </div>

        <div
          onBlur={(event) => {
            const next = event.relatedTarget;
            if (
              next instanceof Node &&
              event.currentTarget.contains(next)
            ) {
              return;
            }
            setFocusedWithinHud(false);
          }}
          onFocus={() => {
            setFocusedWithinHud(true);
          }}
        >
          <ImmersiveControls
            exploring={exploring}
            fullscreen={immersive.fullscreen}
            hud={policyHud}
            onExploreResume={toggleExploreResume}
            onOpenCommands={() => {
              dispatchImmersive({ type: "open-command" });
            }}
            onPlayPause={togglePlayback}
            onToggleFullscreen={() => {
              void toggleFullscreen();
            }}
            onToggleTwin={() => {
              setTwinCompact((compact) => !compact);
            }}
            playbackDisabled={audioConsent === null}
            playing={playing}
          />
          <CausalPath
            beats={causalBeats}
            onSeek={seekBeat}
            timeMs={evaluationTimeMs}
          />
        </div>

        <CommandConstellation
          commands={commands}
          onClose={() => {
            dispatchImmersive({ type: "close-command" });
          }}
          open={immersive.commandOpen}
        />

        <p
          aria-atomic="true"
          aria-live="polite"
          className="aiperf-flow__live-region"
          role="status"
        >
          {liveAnnouncement}
        </p>
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
