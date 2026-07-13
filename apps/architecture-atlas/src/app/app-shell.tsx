// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Link, useRouter } from "@tanstack/react-router";
import {
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
  type AtlasRoutePath,
  type SceneRoute,
} from "../domain/routes";
import {
  executionFlavorSchema,
  type ExecutionFlavor,
} from "../domain/architecture";

interface AppShellProps {
  audience: Audience;
  children: ReactNode;
  sceneRoutes: readonly SceneRoute[];
  activeScenePath: AtlasRoutePath;
  primaryFlavor: ExecutionFlavor;
  compareFlavor: ExecutionFlavor | null;
  graphSearchInputId: string;
  graphSearchQuery: string;
  sharedStateNotice?: string;
  onAudienceChange(audience: Audience): void;
  onPrimaryFlavorChange(flavor: ExecutionFlavor): void;
  onCompareFlavorChange(flavor: ExecutionFlavor | null): void;
  onGraphSearchChange(query: string): void;
  onFitGraph(): void;
  onResetGraph(): void;
  onShareGraphState(): void;
}

function SceneRailLinks({
  routes,
  audience,
  primaryFlavor,
  compareFlavor,
  graphSearchQuery,
}: {
  routes: readonly SceneRoute[];
  audience: Audience;
  primaryFlavor: ExecutionFlavor;
  compareFlavor: ExecutionFlavor | null;
  graphSearchQuery: string;
}) {
  return routes.map(({ path: to, label }) => (
    <Link
      activeOptions={{ exact: to === "/" }}
      activeProps={{ "aria-current": "page" }}
      className="nav-link"
      key={to}
      search={{
        audience,
        primary: primaryFlavor,
        compare: compareFlavor ?? undefined,
        q: graphSearchQuery || undefined,
      }}
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
  sceneRoutes,
  activeScenePath,
  primaryFlavor,
  compareFlavor,
  graphSearchInputId,
  graphSearchQuery,
  sharedStateNotice,
  onAudienceChange,
  onPrimaryFlavorChange,
  onCompareFlavorChange,
  onGraphSearchChange,
  onFitGraph,
  onResetGraph,
  onShareGraphState,
}: AppShellProps) {
  const router = useRouter();
  const [sceneRailCollapsed, setSceneRailCollapsed] = useState(false);

  const handleAudienceChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const result = audienceSchema.safeParse(event.target.value);
    if (result.success) {
      onAudienceChange(result.data);
    }
  };
  const handleGlobalSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const query = graphSearchQuery.trim();
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
    }
  }

  return (
    <div className="app-shell flight-deck-shell">
      <a className="skip-link" href="#atlas-content">
        Skip to content
      </a>

      <aside
        className="side-rail scene-rail"
        data-collapsed={sceneRailCollapsed || undefined}
      >
        <div className="product-mark">
          <span className="product-kicker">AIPerf runtime atlas</span>
          <span className="product-name">Architecture atlas</span>
        </div>
        {sceneRailCollapsed ? null : (
          <nav aria-label="Runtime scenes" className="wide-navigation scene-rail-nav">
            <SceneRailLinks
              audience={audience}
              compareFlavor={compareFlavor}
              graphSearchQuery={graphSearchQuery}
              primaryFlavor={primaryFlavor}
              routes={sceneRoutes}
            />
          </nav>
        )}
        <div className="rail-status legend-strip">
          <span aria-hidden="true" className="status-indicator" />
          <span>Built</span>
          <span aria-hidden="true" className="utility-divider" />
          <span>Planned</span>
        </div>
      </aside>

      <header className="utility-rail command-bar">
        <div className="utility-context scene-context" aria-label="Current scene">
          <span>
            {sceneRoutes.find(({ path }) => path === activeScenePath)?.label ??
              routeCapabilities[0].label}
          </span>
          <span className="utility-divider" aria-hidden="true" />
          <span>Graph-first runtime path</span>
        </div>

        <div className="utility-controls command-controls">
          <form
            aria-label="Compact graph command bar"
            className="global-search command-search"
            onSubmit={handleGlobalSearch}
            role="search"
          >
            <label className="command-field">
              <span>Graph search</span>
              <input
                aria-label="Graph search"
                id={graphSearchInputId}
                onChange={(event) => onGraphSearchChange(event.target.value)}
                placeholder="Component or crate"
                type="search"
                value={graphSearchQuery}
              />
            </label>
          </form>
          <label className="audience-control command-field">
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
          <label className="audience-control command-field">
            <span>Primary flavor</span>
            <select
              aria-label="Primary flavor"
              onChange={(event) => {
                const parsed = executionFlavorSchema.safeParse(event.target.value);
                if (parsed.success) {
                  onPrimaryFlavorChange(parsed.data);
                }
              }}
              value={primaryFlavor}
            >
              {executionFlavorSchema.options.map((flavor) => (
                <option key={flavor} value={flavor}>
                  {flavor}
                </option>
              ))}
            </select>
          </label>
          <label className="audience-control command-field">
            <span>Compare flavor</span>
            <select
              aria-label="Compare flavor"
              onChange={(event) => {
                if (event.target.value === "") {
                  onCompareFlavorChange(null);
                  return;
                }
                const parsed = executionFlavorSchema.safeParse(event.target.value);
                if (parsed.success) {
                  onCompareFlavorChange(parsed.data);
                }
              }}
              value={compareFlavor ?? ""}
            >
              <option value="">None</option>
              {executionFlavorSchema.options.map((flavor) => (
                <option key={flavor} value={flavor}>
                  {flavor}
                </option>
              ))}
            </select>
          </label>
          <button className="command-action" onClick={onFitGraph} type="button">
            Fit graph
          </button>
          <button className="command-action" onClick={onResetGraph} type="button">
            Reset graph
          </button>
          <button
            className="command-action command-action-share"
            onClick={onShareGraphState}
            type="button"
          >
            Share graph state
          </button>
          <button
            className="command-action command-action-toggle"
            onClick={() => setSceneRailCollapsed((collapsed) => !collapsed)}
            type="button"
          >
            {sceneRailCollapsed ? "Expand scene rail" : "Collapse scene rail"}
          </button>
        </div>
      </header>

      {sharedStateNotice ? (
        <p
          aria-label="Graph state recovery notice"
          className="graph-state-notice"
          role="status"
        >
          {sharedStateNotice}
        </p>
      ) : null}

      <main className="content-frame operational-content" id="atlas-content" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
