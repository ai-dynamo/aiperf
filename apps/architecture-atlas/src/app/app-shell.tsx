// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Link } from "@tanstack/react-router";
import {
  useEffect,
  useRef,
  type ChangeEvent,
  type ReactNode,
} from "react";

import {
  audienceSchema,
  type Audience,
} from "../domain/audience";

const navigation = [
  { to: "/", label: "System ownership" },
  { to: "/journey", label: "One-run journey" },
  { to: "/execution", label: "Execution modes" },
  { to: "/data-plane", label: "Data plane" },
  { to: "/observability", label: "Observability" },
  { to: "/parity", label: "Parity ledger" },
  { to: "/atlas", label: "Unified atlas" },
] as const;

interface AppShellProps {
  audience: Audience;
  children: ReactNode;
  presentation: boolean;
  nextRoute?: PresentationRoute;
  previousRoute?: PresentationRoute;
  onExitPresentation(): void;
  onNavigatePresentation(route: PresentationRoute): void;
  onStartPresentation(): void;
  onAudienceChange(audience: Audience): void;
}

type PresentationRoute =
  | "/"
  | "/journey"
  | "/execution"
  | "/data-plane"
  | "/observability"
  | "/parity";

function NavigationLinks() {
  return navigation.map(({ to, label }) => (
    <Link
      activeOptions={{ exact: to === "/" }}
      activeProps={{ "aria-current": "page" }}
      className="nav-link"
      key={to}
      search={(previous) => previous}
      to={to}
    >
      <span className="nav-marker" aria-hidden="true" />
      {label}
    </Link>
  ));
}

export function AppShell({
  audience,
  children,
  presentation,
  nextRoute,
  previousRoute,
  onExitPresentation,
  onNavigatePresentation,
  onStartPresentation,
  onAudienceChange,
}: AppShellProps) {
  const presentationMain = useRef<HTMLElement>(null);
  const nextControl = useRef<HTMLButtonElement>(null);
  const previousControl = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (presentation) {
      presentationMain.current?.focus();
    }
  }, [presentation]);

  useEffect(() => {
    if (!presentation) {
      return undefined;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onExitPresentation();
      } else if (event.key === "ArrowLeft" && previousRoute) {
        previousControl.current?.click();
      } else if (event.key === "ArrowRight" && nextRoute) {
        nextControl.current?.click();
      } else {
        return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    nextRoute,
    onExitPresentation,
    onNavigatePresentation,
    presentation,
    previousRoute,
  ]);

  const handleAudienceChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const result = audienceSchema.safeParse(event.target.value);
    if (result.success) {
      onAudienceChange(result.data);
    }
  };

  if (presentation) {
    return (
      <div className="app-shell presentation-shell">
        <main
          className="content-frame"
          id="atlas-content"
          ref={presentationMain}
          tabIndex={-1}
        >
          {children}
        </main>
        <nav
          aria-label="Presentation routes"
          className="presentation-navigation"
        >
          {previousRoute ? (
            <button
              onClick={() => onNavigatePresentation(previousRoute)}
              ref={previousControl}
              type="button"
            >
              Previous
            </button>
          ) : (
            <span aria-disabled="true">Previous</span>
          )}
          <span className="presentation-lens">{audience}</span>
          <button onClick={onExitPresentation} type="button">
            Exit presentation
          </button>
          {nextRoute ? (
            <button
              onClick={() => onNavigatePresentation(nextRoute)}
              ref={nextControl}
              type="button"
            >
              Next
            </button>
          ) : (
            <span aria-disabled="true">Next</span>
          )}
        </nav>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#atlas-content">
        Skip to content
      </a>

      <aside className="side-rail">
        <div className="product-mark">
          <span className="product-kicker">AIPerf systems observatory</span>
          <span className="product-name">Architecture atlas</span>
        </div>
        <nav aria-label="Architecture views" className="wide-navigation">
          <NavigationLinks />
        </nav>
        <div className="rail-status">
          <span className="status-indicator" aria-hidden="true" />
          Foundation dataset
        </div>
      </aside>

      <header className="utility-rail">
        <details className="compact-navigation">
          <summary>Explore</summary>
          <nav aria-label="Compact architecture views">
            <NavigationLinks />
          </nav>
        </details>

        <div className="utility-context" aria-label="Current context">
          <span>Atlas foundation</span>
          <span className="utility-divider" aria-hidden="true" />
          <span>Source-grounded views</span>
        </div>

        <div className="utility-controls">
          <label className="audience-control">
            <span>Audience</span>
            <select
              aria-label="Audience"
              value={audience}
              onChange={handleAudienceChange}
            >
              <option value="executive">Executive</option>
              <option value="developer">Developer</option>
              <option value="maintainer">Maintainer</option>
            </select>
          </label>
          <button
            className="presentation-control"
            onClick={onStartPresentation}
            type="button"
          >
            Present this view
          </button>
        </div>
      </header>

      <main className="content-frame" id="atlas-content" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
