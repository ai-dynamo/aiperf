// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Link } from "@tanstack/react-router";
import type { ChangeEvent, ReactNode } from "react";

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
  onAudienceChange(audience: Audience): void;
}

function NavigationLinks() {
  return navigation.map(({ to, label }) => (
    <Link
      activeOptions={{ exact: to === "/" }}
      activeProps={{ "aria-current": "page" }}
      className="nav-link"
      key={to}
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
  onAudienceChange,
}: AppShellProps) {
  const handleAudienceChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const result = audienceSchema.safeParse(event.target.value);
    if (result.success) {
      onAudienceChange(result.data);
    }
  };

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
            <select value={audience} onChange={handleAudienceChange}>
              <option value="executive">Executive</option>
              <option value="developer">Developer</option>
              <option value="maintainer">Maintainer</option>
            </select>
          </label>
          <button
            className="presentation-control"
            disabled
            title="Presentation controls arrive with atlas content"
            type="button"
          >
            Presentation controls
          </button>
        </div>
      </header>

      <main className="content-frame" id="atlas-content" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
