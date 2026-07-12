// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import type { RouterHistory } from "@tanstack/react-router";
import { useEffect } from "react";

import { AppShell } from "../app/app-shell";
import {
  parseAudienceSearch,
  persistAudience,
  resolveAudience,
  type Audience,
} from "../domain/audience";
import { RoutePlaceholder } from "./placeholders";

const unavailableAudienceStorage = {
  getItem: () => null,
  setItem: () => undefined,
};

function getAudienceStorage() {
  try {
    return window.localStorage;
  } catch {
    return unavailableAudienceStorage;
  }
}

function RootRouteComponent() {
  const search = rootRoute.useSearch();
  const navigate = rootRoute.useNavigate();
  const storage = getAudienceStorage();
  const audience = resolveAudience(search.audience, storage);

  useEffect(() => {
    persistAudience(audience, storage);
    if (search.audience !== audience) {
      void navigate({
        replace: true,
        search: (previous) => ({ ...previous, audience }),
      });
    }
  }, [audience, navigate, search.audience, storage]);

  const handleAudienceChange = (nextAudience: Audience) => {
    persistAudience(nextAudience, storage);
    void navigate({
      replace: true,
      search: (previous) => ({ ...previous, audience: nextAudience }),
    });
  };

  return (
    <AppShell
      audience={audience}
      onAudienceChange={handleAudienceChange}
    >
      <Outlet />
    </AppShell>
  );
}

const rootRoute = createRootRoute({
  component: RootRouteComponent,
  validateSearch: parseAudienceSearch,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => (
    <RoutePlaceholder
      eyebrow="View 01 / Product boundary"
      title="System ownership"
      summary="Trace the division of responsibility between Python authoring, the Rust runner, external systems, and compatibility surfaces."
    />
  ),
});

const journeyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/journey",
  component: () => (
    <RoutePlaceholder
      eyebrow="View 02 / One run"
      title="One-run journey"
      summary="Follow one authored configuration through capability preflight, protocol v2 execution, native reporting, and Python presentation."
    />
  ),
});

const executionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/execution",
  component: () => (
    <RoutePlaceholder
      eyebrow="View 03 / Runtime"
      title="Execution modes"
      summary="Compare the clock, transport, scheduling, and lifecycle seams shared by online and feature-gated offline execution."
    />
  ),
});

const dataPlaneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-plane",
  component: () => (
    <RoutePlaceholder
      eyebrow="View 04 / Request shaping"
      title="Data plane"
      summary="Inspect how datasets, segments, endpoint preparation, Graph-IR, and generated media become dispatchable requests."
    />
  ),
});

const observabilityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/observability",
  component: () => (
    <RoutePlaceholder
      eyebrow="View 05 / Measurement"
      title="Observability and evaluation"
      summary="Connect request observations to native metrics, telemetry, accuracy, and provider-neutral evaluation."
    />
  ),
});

const parityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/parity",
  component: () => (
    <RoutePlaceholder
      eyebrow="View 06 / Migration state"
      title="Parity ledger"
      summary="Separate built, conditional, compatibility-only, legacy-parallel, and unbuilt surfaces."
    />
  ),
});

const atlasRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/atlas",
  component: () => (
    <RoutePlaceholder
      eyebrow="Unified view / System map"
      title="Unified architecture atlas"
      summary="Navigate the full ownership, execution, data, and measurement topology from one source-grounded map."
    />
  ),
});

const crateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/crates/$crateId",
  component: CrateRouteComponent,
});

function CrateRouteComponent() {
  const { crateId } = crateRoute.useParams();
  return (
    <RoutePlaceholder
      eyebrow="Maintainer reference / Crate"
      title={crateId}
      summary="Crate ownership, contracts, dependencies, and source evidence will be populated from the validated atlas dataset."
    />
  );
}

const routeTree = rootRoute.addChildren([
  indexRoute,
  journeyRoute,
  executionRoute,
  dataPlaneRoute,
  observabilityRoute,
  parityRoute,
  atlasRoute,
  crateRoute,
]);

interface CreateAppRouterOptions {
  history?: RouterHistory;
}

export function createAppRouter(options: CreateAppRouterOptions = {}) {
  return createRouter({
    routeTree,
    history: options.history,
    defaultPreload: "intent",
  });
}

export type AppRouter = ReturnType<typeof createAppRouter>;

declare module "@tanstack/react-router" {
  interface Register {
    router: AppRouter;
  }
}
