// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Link, useRouter } from "@tanstack/react-router";
import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type ReactNode,
} from "react";

import { architectureCatalog } from "../content";
import {
  audienceSchema,
  type Audience,
} from "../domain/audience";
import {
  routeCapabilities,
  routeSupports,
  type PresentationRoute,
} from "../domain/routes";

interface AppShellProps {
  audience: Audience;
  children: ReactNode;
  presentation: boolean;
  presentationAvailable: boolean;
  nextRoute?: PresentationRoute;
  previousRoute?: PresentationRoute;
  onExitPresentation(): void;
  onNavigatePresentation(route: PresentationRoute): void;
  onStartPresentation(): void;
  onAudienceChange(audience: Audience): void;
}

function NavigationLinks() {
  return routeCapabilities.map(({ path: to, label }) => (
    <Link
      activeOptions={{ exact: to === "/" }}
      activeProps={{ "aria-current": "page" }}
      className="nav-link"
      key={to}
      search={(previous) => ({
        audience: previous.audience,
        modes:
          routeSupports(to, "filters") && !routeSupports(to, "atlasState")
            ? previous.modes
            : undefined,
        statuses:
          routeSupports(to, "filters") && !routeSupports(to, "atlasState")
            ? previous.statuses
            : undefined,
      })}
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
  presentationAvailable,
  nextRoute,
  previousRoute,
  onExitPresentation,
  onNavigatePresentation,
  onStartPresentation,
  onAudienceChange,
}: AppShellProps) {
  const presentationMain = useRef<HTMLElement>(null);
  const presentationEntry = useRef<HTMLButtonElement>(null);
  const nextControl = useRef<HTMLButtonElement>(null);
  const previousControl = useRef<HTMLButtonElement>(null);
  const wasPresenting = useRef(false);
  const router = useRouter();
  const [globalQuery, setGlobalQuery] = useState("");

  useEffect(() => {
    if (presentation) {
      presentationMain.current?.focus();
    } else if (wasPresenting.current) {
      presentationEntry.current?.focus();
    }
    wasPresenting.current = presentation;
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
  const handleGlobalSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const query = globalQuery.trim();
    if (!query) {
      return;
    }
    const crate = architectureCatalog.crates.find(
      ({ packageName }) =>
        packageName.toLocaleLowerCase() === query.toLocaleLowerCase(),
    );
    if (crate) {
      void router.navigate({
        to: "/crates/$crateId",
        params: { crateId: crate.packageName },
        search: { audience },
      });
    } else {
      void router.navigate({
        to: "/atlas",
        search: { audience, query },
      });
    }
    setGlobalQuery("");
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
          <form
            aria-label="Global architecture search"
            className="global-search"
            onSubmit={handleGlobalSearch}
            role="search"
          >
            <label>
              <span>Search architecture</span>
              <input
                aria-label="Search architecture"
                onChange={(event) => setGlobalQuery(event.target.value)}
                placeholder="Component or crate"
                type="search"
                value={globalQuery}
              />
            </label>
          </form>
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
          {presentationAvailable ? (
            <button
              className="presentation-control"
              onClick={onStartPresentation}
              ref={presentationEntry}
              type="button"
            >
              Present this view
            </button>
          ) : null}
        </div>
      </header>

      <main className="content-frame" id="atlas-content" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
