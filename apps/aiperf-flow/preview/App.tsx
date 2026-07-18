// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowIr } from "../packages/schema/src/ir";
import React, {
  type ReactNode,
  type RefObject,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { FlowApp } from "../packages/runtime/src/app";
import type { KokoroNarratorSnapshot } from "../packages/runtime/src/narrative/kokoro-narrator";
import { COMPILED_EXPLAINER_DECKS } from "../packages/runtime/src/explainer/compiled-decks";

import {
  previewWorkspace,
  type PreviewWorkspace,
  discoverScenesByFlow,
} from "./fixture";
import { HomePage } from "./home-page";
import { ExplainerDeckPicker } from "./explainer-deck-picker";
import { ExplainerDeckNavigator } from "./explainer-deck-navigator";
import {
  createPreviewNarratorBackend,
  prewarmPreviewNarrator,
  subscribePreviewKokoroState,
} from "./narrator-backend";

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

type Theme = "systems-chalk" | "legacy" | "core";
type AudioConsent = "yes" | "no" | "unset";

const THEME_STORAGE_KEY = "aiperf-flow-theme";
const AUDIO_CONSENT_KEY = "aiperf-flow-audio-consent";

function loadThemeFromStorage(): Theme {
  if (typeof localStorage === "undefined") {
    return "systems-chalk";
  }
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "legacy" || stored === "core") {
      return stored;
    }
  } catch {
    // Ignore storage errors
  }
  return "systems-chalk";
}

function saveThemeToStorage(theme: Theme): void {
  if (typeof localStorage === "undefined") {
    return;
  }
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Ignore storage errors
  }
}

function loadAudioConsentFromStorage(): AudioConsent {
  if (typeof localStorage === "undefined") {
    return "unset";
  }
  try {
    const stored = localStorage.getItem(AUDIO_CONSENT_KEY);
    if (stored === "yes" || stored === "no") {
      return stored;
    }
  } catch {
    // Ignore storage errors
  }
  return "unset";
}

function saveAudioConsentToStorage(consent: AudioConsent): void {
  if (typeof localStorage === "undefined") {
    return;
  }
  try {
    localStorage.setItem(AUDIO_CONSENT_KEY, consent);
  } catch {
    // Ignore storage errors
  }
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

function flowWithNarrowScene(flow: FlowIr, sceneId: string): FlowIr {
  return {
    ...flow,
    scenes: flow.scenes.map((scene) => {
      if (scene.id !== sceneId) {
        return scene;
      }
      const narrow = scene.responsive.find(
        (variant) => variant.condition === "(max-width: 860px)",
      );
      return narrow === undefined ? scene : { ...scene, roots: narrow.roots };
    }),
  };
}

type BreadcrumbProps = Readonly<{
  workspace: PreviewWorkspace;
  activeFlowId: string;
  activeSceneId: string;
  onSelectScene(flowId: string, sceneId: string): void;
}>;

function Breadcrumb({
  workspace,
  activeFlowId,
  activeSceneId,
  onSelectScene,
}: BreadcrumbProps): ReactNode {
  const activeFlow = workspace.flows[activeFlowId] ?? workspace.flow;
  const sceneIndex = activeFlow.scenes.findIndex((s) => s.id === activeSceneId);
  const stepNumber = sceneIndex >= 0 ? sceneIndex + 1 : 1;

  return (
    <nav className="preview-breadcrumb" aria-label="Scene progression">
      <div className="breadcrumb-steps">
        {activeFlow.scenes.map((scene, index) => {
          const stepNum = index + 1;
          const isActive = scene.id === activeSceneId;
          return (
            <button
              key={scene.id}
              className={`breadcrumb-step ${isActive ? "is-active" : ""}`}
              onClick={() => onSelectScene(activeFlowId, scene.id)}
              type="button"
              aria-current={isActive ? "step" : undefined}
            >
              <span className="breadcrumb-number">{stepNum}</span>
              <span className="breadcrumb-title">{scene.title}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

type BottomNavProps = Readonly<{
  workspace: PreviewWorkspace;
  activeFlowId: string;
  activeSceneId: string;
  onSelectScene(flowId: string, sceneId: string): void;
}>;

function BottomNav({
  workspace,
  activeFlowId,
  activeSceneId,
  onSelectScene,
}: BottomNavProps): ReactNode {
  const activeFlow = workspace.flows[activeFlowId] ?? workspace.flow;
  const sceneIndex = activeFlow.scenes.findIndex((s) => s.id === activeSceneId);
  const canGoPrev = sceneIndex > 0;
  const canGoNext = sceneIndex < activeFlow.scenes.length - 1;

  function goToPrev(): void {
    if (canGoPrev) {
      const prevScene = activeFlow.scenes[sceneIndex - 1];
      if (prevScene !== undefined) {
        onSelectScene(activeFlowId, prevScene.id);
      }
    }
  }

  function goToNext(): void {
    if (canGoNext) {
      const nextScene = activeFlow.scenes[sceneIndex + 1];
      if (nextScene !== undefined) {
        onSelectScene(activeFlowId, nextScene.id);
      }
    }
  }

  return (
    <nav className="preview-bottom-nav" aria-label="Scene navigation">
      <button
        onClick={goToPrev}
        disabled={!canGoPrev}
        type="button"
        className="nav-button nav-back"
      >
        ← Back
      </button>
      <button
        onClick={goToNext}
        disabled={!canGoNext}
        type="button"
        className="nav-button nav-next"
      >
        Next →
      </button>
    </nav>
  );
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
      aria-label="Flow browser"
      className="flow-browser"
    >
      <div className="flow-browser-head">
        <div>
          <span className="flow-browser-kicker">Workspace</span>
          <strong>Flows</strong>
        </div>
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
  const scenesByFlow = useMemo(() => discoverScenesByFlow(), []);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [showHome, setShowHome] = useState(true);
  const [activeFlowId, setActiveFlowId] = useState(
    workspace.navigation.active.flowId,
  );
  const [activeSceneId, setActiveSceneId] = useState(
    workspace.navigation.active.sceneId,
  );
  const [reducedMotion, setReducedMotion] = useState(false);
  const [narrowLayout, setNarrowLayout] = useState(false);
  const [kokoroState, setKokoroState] = useState<KokoroNarratorSnapshot | null>(
    null,
  );
  const [theme, setTheme] = useState<Theme>(() => loadThemeFromStorage());
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const [audioConsent, setAudioConsent] = useState<AudioConsent>("unset");
  const [hasLeftSite, setHasLeftSite] = useState(false);
  const [selectedExplainerDeckId, setSelectedExplainerDeckId] = useState<string | null>(null);
  const [explainerSlideIndex, setExplainerSlideIndex] = useState(0);
  const [showExplainerPicker, setShowExplainerPicker] = useState(false);
  const flow = useMemo(() => {
    const base = workspace.flows[activeFlowId] ?? workspace.flow;
    const responsive = narrowLayout
      ? flowWithNarrowScene(base, activeSceneId)
      : base;
    return flowWithActiveSceneFirst(responsive, activeSceneId);
  }, [workspace, activeFlowId, activeSceneId, narrowLayout]);

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

  useEffect(() => {
    const media = window.matchMedia("(max-width: 860px)");
    const sync = (): void => setNarrowLayout(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    saveThemeToStorage(theme);
  }, [theme]);

  useEffect(() => {
    const handleVisibilityChange = (): void => {
      if (document.hidden) {
        setHasLeftSite(true);
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  useEffect(() => {
    if (!showThemeMenu) {
      return;
    }
    const handleClickOutside = (event: MouseEvent): void => {
      const target = event.target as Node;
      // Check if click is outside the theme menu area
      if (!target || !(target instanceof Element)) {
        return;
      }
      if (
        !target.closest("[data-theme-menu]") &&
        !target.closest("button[aria-label='Theme selector']")
      ) {
        setShowThemeMenu(false);
      }
    };
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [showThemeMenu]);

  function selectScene(flowId: string, sceneId: string): void {
    setActiveFlowId(flowId);
    setActiveSceneId(sceneId);
    setShowHome(false);
  }

  function goHome(): void {
    setShowHome(true);
  }

  function openExplainerPicker(): void {
    setShowExplainerPicker(true);
    setShowHome(false);
  }

  function selectExplainerDeck(deckId: string): void {
    setSelectedExplainerDeckId(deckId);
    setExplainerSlideIndex(0);
    setShowExplainerPicker(false);
  }

  function closeExplainerDeck(): void {
    setSelectedExplainerDeckId(null);
    setExplainerSlideIndex(0);
    setShowExplainerPicker(false);
    setShowHome(true);
  }

  function handleExplainerSlideChange(newIndex: number): void {
    setExplainerSlideIndex(newIndex);
  }

  function handleThemeChange(newTheme: Theme): void {
    setTheme(newTheme);
    setShowThemeMenu(false);
  }

  function cycleTheme(): void {
    const themes: Theme[] = ["systems-chalk", "legacy", "core"];
    const currentIndex = themes.indexOf(theme);
    const nextIndex = (currentIndex + 1) % themes.length;
    handleThemeChange(themes[nextIndex]!);
  }

  const voiceStatus =
    kokoroState?.status === "loading"
      ? `Loading voice ${Math.round((kokoroState.progress ?? 0) * 100)}%`
      : kokoroState?.status === "needs-user-activation"
        ? "Press play for voice"
        : null;

  const themeLabel = theme
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

  return (
    <div
      className="preview-shell"
      data-preview-layout={
        activeSceneId === "request-investigation" ? "hub-spoke" : "standard"
      }
    >
      <header className="preview-topbar">
        <div className="preview-brand-cluster">
          <button
            className="preview-study"
            aria-label="AIPerf Flow home"
            onClick={goHome}
            type="button"
            style={{
              background: "none",
              border: "none",
              padding: 0,
              cursor: "pointer",
              textAlign: "left",
            }}
          >
            <span>
              <small>AIPerf Flow · Scene study 02</small>
              <h1>From one request to the whole system</h1>
            </span>
          </button>
        </div>
        <div
          className="preview-theme-cluster"
          style={{ display: showHome ? "none" : "flex" }}
        >
          {voiceStatus === null ? null : (
            <p className="preview-status" aria-live="polite">
              {voiceStatus}
            </p>
          )}
          <div style={{ position: "relative" }}>
            <button
              aria-label="Theme selector"
              aria-expanded={showThemeMenu}
              onClick={() => setShowThemeMenu((prev) => !prev)}
              type="button"
              style={{
                minHeight: "2rem",
                padding: "0.4rem 0.6rem",
                border: "1px solid var(--preview-guide)",
                borderRadius: "6px",
                background: "var(--preview-control)",
                color: "var(--preview-muted)",
                cursor: "pointer",
                fontSize: "0.7rem",
                fontWeight: 600,
                textTransform: "uppercase",
              }}
            >
              {themeLabel}
            </button>
            {showThemeMenu && (
              <div
                data-theme-menu="true"
                style={{
                  position: "absolute",
                  top: "100%",
                  right: 0,
                  marginTop: "0.5rem",
                  background: "var(--preview-panel)",
                  border: "1px solid var(--preview-guide)",
                  borderRadius: "6px",
                  minWidth: "120px",
                  zIndex: 1000,
                  boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
                }}
              >
                {(["systems-chalk", "legacy", "core"] as const).map(
                  (themeOption) => (
                    <button
                      key={themeOption}
                      onClick={() => handleThemeChange(themeOption)}
                      type="button"
                      style={{
                        display: "block",
                        width: "100%",
                        padding: "0.5rem 0.75rem",
                        border: "none",
                        background:
                          theme === themeOption
                            ? "var(--preview-signal)"
                            : "transparent",
                        color:
                          theme === themeOption
                            ? "var(--preview-board)"
                            : "var(--preview-chalk)",
                        textAlign: "left",
                        cursor: "pointer",
                        fontSize: "0.7rem",
                        fontWeight: 500,
                        textTransform: "capitalize",
                      }}
                    >
                      {themeOption.split("-").join(" ")}
                    </button>
                  ),
                )}
              </div>
            )}
          </div>
          <button
            aria-label="Toggle theme"
            onClick={cycleTheme}
            title="Click to cycle through themes"
            type="button"
            style={{
              minHeight: "2rem",
              padding: "0.4rem 0.6rem",
              border: "1px solid var(--preview-guide)",
              borderRadius: "6px",
              background: "var(--preview-control)",
              color: "var(--preview-muted)",
              cursor: "pointer",
              fontSize: "0.7rem",
              fontWeight: 600,
            }}
          >
            ⟳
          </button>
        </div>
      </header>

      <div
        className="flow-workspace"
      >
        <DocumentBrowser
          activeFlowId={activeFlowId}
          activeSceneId={activeSceneId}
          onSelectScene={selectScene}
          searchRef={searchRef}
          workspace={workspace}
        />

        <div className="flow-main-section">
          <Breadcrumb
            workspace={workspace}
            activeFlowId={activeFlowId}
            activeSceneId={activeSceneId}
            onSelectScene={selectScene}
          />

          <main
            className="runtime-story"
            data-theme={theme}
            style={{
              ...(theme === "legacy" && {
                "--flow-board": "#1a1a1a",
                "--flow-panel": "#222",
                "--flow-raised": "#2a2a2a",
                "--flow-control-surface": "#2a2a2a",
                "--flow-chalk": "#e8e8e8",
                "--flow-chalk-muted": "#999",
              } as React.CSSProperties),
              ...(theme === "core" && {
                "--flow-board": "#0d1117",
                "--flow-panel": "#161b22",
                "--flow-raised": "#21262d",
                "--flow-control-surface": "#21262d",
                "--flow-chalk": "#f0f6fc",
                "--flow-chalk-muted": "#8b949e",
              } as React.CSSProperties),
            }}
          >
            {selectedExplainerDeckId ? (
              <ExplainerDeckNavigator
                deckId={selectedExplainerDeckId}
                slideIndex={explainerSlideIndex}
                onSlideChange={handleExplainerSlideChange}
                onBackClick={closeExplainerDeck}
              />
            ) : (
              <FlowApp
                key={`${activeFlowId}:${activeSceneId}`}
                flow={flow}
                narratorBackend={narratorBackend}
                reducedMotion={reducedMotion}
                requireAudioConsent={audioConsent === "unset"}
                onAudioConsentChange={(hasConsented) => {
                  setAudioConsent(hasConsented ? "yes" : "no");
                }}
                autoPlay={audioConsent === "yes"}
              />
            )}
          </main>

          <BottomNav
            workspace={workspace}
            activeFlowId={activeFlowId}
            activeSceneId={activeSceneId}
            onSelectScene={selectScene}
          />
        </div>
      </div>

      {showHome && (
        <HomePage
          scenesByFlow={scenesByFlow}
          onSelectScene={selectScene}
          onOpenExplainers={openExplainerPicker}
        />
      )}
      {showExplainerPicker && !selectedExplainerDeckId && (
        <ExplainerDeckPicker
          decks={COMPILED_EXPLAINER_DECKS}
          onDeckSelect={selectExplainerDeck}
        />
      )}
      {!showHome && activeSceneId === "request-investigation" ? (
        <footer className="preview-legend">
          <div aria-label="Semantic legend" className="preview-legend-items">
            <span data-legend="cause">active cause</span>
            <span data-legend="request">selected request</span>
            <span data-legend="decision">decision point</span>
          </div>
          <span>Entity → connector → destination → annotation</span>
        </footer>
      ) : null}
    </div>
  );
}
