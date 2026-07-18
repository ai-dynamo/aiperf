// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from "react";

import { COMPILED_EXPLAINER_DECKS } from "../packages/runtime/src/explainer/compiled-decks";

import { ExplainerDeckPicker } from "./explainer-deck-picker";
import { ExplainerDeckNavigator } from "./explainer-deck-navigator";

export { unlockPreviewSpeech } from "./narrator-backend";

type Theme = "systems-chalk" | "legacy" | "core";

const THEME_STORAGE_KEY = "aiperf-flow-theme";

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

/** Preview host: explainer deck picker + slide navigator. */
export function App() {
  const [theme, setTheme] = useState<Theme>(() => loadThemeFromStorage());
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const [selectedExplainerDeckId, setSelectedExplainerDeckId] = useState<
    string | null
  >(null);
  const [explainerSlideIndex, setExplainerSlideIndex] = useState(0);

  useEffect(() => {
    saveThemeToStorage(theme);
  }, [theme]);

  useEffect(() => {
    if (!showThemeMenu) {
      return;
    }
    const handleClickOutside = (event: MouseEvent): void => {
      const target = event.target as Node;
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

  function selectExplainerDeck(deckId: string): void {
    setSelectedExplainerDeckId(deckId);
    setExplainerSlideIndex(0);
  }

  function closeExplainerDeck(): void {
    setSelectedExplainerDeckId(null);
    setExplainerSlideIndex(0);
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

  const themeLabel = theme
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

  const selectedDeck = selectedExplainerDeckId
    ? COMPILED_EXPLAINER_DECKS.find((d) => d.id === selectedExplainerDeckId)
    : undefined;

  return (
    <div className="preview-shell" data-preview-layout="standard">
      <header className="preview-topbar">
        <div className="preview-brand-cluster">
          <span>
            <small>AIPerf Flow · Explainers</small>
            <h1>From one request to the whole system</h1>
          </span>
        </div>
        <div className="preview-theme-cluster" style={{ display: "flex" }}>
          <div style={{ position: "relative" }}>
            <button
              aria-label="Theme selector"
              aria-expanded={showThemeMenu}
              onClick={() => setShowThemeMenu((prev) => !prev)}
              type="button"
              style={{
                padding: "0.5rem 1rem",
                border: "1px solid var(--preview-guide)",
                borderRadius: "4px",
                background: "var(--preview-control)",
                color: "var(--preview-chalk)",
                cursor: "pointer",
                fontSize: "0.9rem",
                fontWeight: 500,
                textTransform: "none",
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
                  borderRadius: "4px",
                  minWidth: "140px",
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
                        padding: "0.5rem 1rem",
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
                        fontSize: "0.9rem",
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
              padding: "0.5rem 1rem",
              border: "1px solid var(--preview-guide)",
              borderRadius: "4px",
              background: "var(--preview-control)",
              color: "var(--preview-chalk)",
              cursor: "pointer",
              fontSize: "0.9rem",
              fontWeight: 500,
            }}
          >
            ⟳
          </button>
        </div>
      </header>

      <main
        className="runtime-story"
        data-theme={theme}
        style={{
          height: "100%",
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
        {selectedDeck ? (
          <ExplainerDeckNavigator
            deckId={selectedExplainerDeckId!}
            deck={selectedDeck}
            slideIndex={explainerSlideIndex}
            onSlideChange={handleExplainerSlideChange}
            onBackClick={closeExplainerDeck}
          />
        ) : (
          <ExplainerDeckPicker
            decks={COMPILED_EXPLAINER_DECKS}
            onDeckSelect={selectExplainerDeck}
          />
        )}
      </main>
    </div>
  );
}
