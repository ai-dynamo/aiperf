// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowIr, SceneIr } from "../packages/schema/src/ir";
import {
  type ReactNode,
  type RefObject,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { FlowApp } from "../packages/runtime/src/app";
import type { KokoroNarratorSnapshot } from "../packages/runtime/src/narrative/kokoro-narrator";
import type { NarrativeCue } from "../packages/runtime/src/narrative/timeline";

import {
  previewDurationMs,
  previewWorkspace,
  type PreviewWorkspace,
} from "./fixture";
import {
  createPreviewNarratorBackend,
  prewarmPreviewNarrator,
  subscribePreviewKokoroState,
} from "./narrator-backend";

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

/**
 * Places the selected scene first so shared {@link FlowApp} mounts on the
 * browser selection without owning a second scene index.
 */
function flowWithActiveSceneFirst(flow: FlowIr, sceneId: string): FlowIr {
  const index = flow.scenes.findIndex((entry) => entry.id === sceneId);
  if (index <= 0) {
    return flow;
  }
  const scenes = [...flow.scenes];
  const [active] = scenes.splice(index, 1);
  if (active === undefined) {
    return flow;
  }
  return { ...flow, scenes: [active, ...scenes] };
}

type DocumentBrowserProps = Readonly<{
  workspace: PreviewWorkspace;
  activeFlowId: string;
  activeSceneId: string;
  collapsed: boolean;
  searchRef: RefObject<HTMLInputElement | null>;
  onCollapse(): void;
  onSelectScene(flowId: string, sceneId: string): void;
}>;

function DocumentBrowser({
  workspace,
  activeFlowId,
  activeSceneId,
  collapsed,
  searchRef,
  onCollapse,
  onSelectScene,
}: DocumentBrowserProps): ReactNode {
  const totalScenes = workspace.navigation.files.reduce(
    (sum, file) =>
      sum +
      file.chapters.reduce(
        (chapterSum, chapter) => chapterSum + chapter.scenes.length,
        0,
      ),
    0,
  );

  return (
    <aside
      aria-hidden={collapsed}
      aria-label="Flow browser"
      className="flow-browser"
      inert={collapsed || undefined}
    >
      <div className="flow-browser-head">
        <div>
          <span className="flow-browser-kicker">Workspace</span>
          <strong>Flows</strong>
        </div>
        <button
          aria-label="Collapse Flow browser"
          onClick={onCollapse}
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
          ref={searchRef}
          type="search"
        />
        <kbd>⌘K</kbd>
      </label>

      <nav className="flow-tree" aria-label="Flow files and scenes">
        {workspace.navigation.files.map((file) => {
          const open = file.id === activeFlowId;
          return (
            <details key={file.id} open={open}>
              <summary>
                <span className="flow-file-mark" aria-hidden="true">
                  ◆
                </span>
                <span>{file.sourceName}</span>
                <small>
                  {file.chapters.reduce(
                    (sum, chapter) => sum + chapter.scenes.length,
                    0,
                  )}
                </small>
              </summary>
              <div className="flow-tree-branch">
                {file.chapters.map((chapter) => (
                  <section key={chapter.id}>
                    <p>{chapter.name}</p>
                    <ul>
                      {chapter.scenes.map((entry) => {
                        const active =
                          file.id === activeFlowId &&
                          entry.id === activeSceneId;
                        return (
                          <li key={entry.id}>
                            <button
                              aria-current={active ? "page" : undefined}
                              className={active ? "is-active" : undefined}
                              onClick={() => onSelectScene(file.id, entry.id)}
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
                ))}
              </div>
            </details>
          );
        })}
      </nav>

      <footer className="flow-browser-foot">
        <span className="preview-status-dot" />
        {workspace.navigation.files.length} flow
        {workspace.navigation.files.length === 1 ? "" : "s"} · {totalScenes}{" "}
        scene{totalScenes === 1 ? "" : "s"}
      </footer>
    </aside>
  );
}

/** Preview host: document browser overlay around shared {@link FlowApp}. */
export function App() {
  const workspace = useMemo(() => previewWorkspace(), []);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [browserCollapsed, setBrowserCollapsed] = useState(true);
  const [activeFlowId, setActiveFlowId] = useState(
    workspace.navigation.active.flowId,
  );
  const [activeSceneId, setActiveSceneId] = useState(
    workspace.navigation.active.sceneId,
  );
  const [reducedMotion, setReducedMotion] = useState(false);
  const [kokoroState, setKokoroState] = useState<KokoroNarratorSnapshot | null>(
    null,
  );
  const flow = useMemo(() => {
    const base = workspace.flows[activeFlowId] ?? workspace.flow;
    return flowWithActiveSceneFirst(base, activeSceneId);
  }, [workspace, activeFlowId, activeSceneId]);

  const narratorBackend = useMemo(
    () => createPreviewNarratorBackend(),
    [],
  );

  useEffect(() => {
    prewarmPreviewNarrator();
    return subscribePreviewKokoroState((state) => {
      setKokoroState((previous) =>
        previous !== null &&
        previous.status === state.status &&
        previous.engine === state.engine &&
        previous.progress === state.progress &&
        previous.progressFile === state.progressFile &&
        previous.error === state.error &&
        previous.activeCueId === state.activeCueId &&
        previous.needsUserActivation === state.needsUserActivation
          ? previous
          : state,
      );
    });
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = (): void => setReducedMotion(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  function selectScene(flowId: string, sceneId: string): void {
    setActiveFlowId(flowId);
    setActiveSceneId(sceneId);
  }

  const voiceStatus =
    kokoroState?.status === "loading"
      ? `Loading voice ${Math.round((kokoroState.progress ?? 0) * 100)}%`
      : kokoroState?.status === "needs-user-activation"
        ? "Press play for voice"
        : null;

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
        {voiceStatus === null ? null : (
          <p className="preview-status" aria-live="polite">
            {voiceStatus}
          </p>
        )}
      </header>

      <div
        className="flow-workspace"
        data-browser-collapsed={browserCollapsed ? "true" : "false"}
      >
        <DocumentBrowser
          activeFlowId={activeFlowId}
          activeSceneId={activeSceneId}
          collapsed={browserCollapsed}
          onCollapse={() => setBrowserCollapsed(true)}
          onSelectScene={selectScene}
          searchRef={searchRef}
          workspace={workspace}
        />

        <main className="runtime-story">
          <FlowApp
            key={`${activeFlowId}:${activeSceneId}`}
            flow={flow}
            narratorBackend={narratorBackend}
            reducedMotion={reducedMotion}
            requireAudioConsent
          />
        </main>
      </div>
    </div>
  );
}
