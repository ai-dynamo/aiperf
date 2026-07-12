// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
  useRouter,
  useRouterState,
} from "@tanstack/react-router";
import type { RouterHistory } from "@tanstack/react-router";
import { lazy, Suspense, useEffect } from "react";

import { AudienceProvider, useAudience } from "../app/audience-context";
import { AppShell } from "../app/app-shell";
import {
  persistAudience,
  resolveAudience,
  type Audience,
} from "../domain/audience";
import type {
  ArchitectureStatus,
  ExecutionMode,
} from "../domain/architecture";
import {
  presentationRoutePaths,
  routeSupports,
  type GuidedRoute,
} from "../domain/routes";
import {
  encodeSelection,
  parseAtlasSearch,
  parseModes,
  parseOwnership,
  parseStatuses,
} from "../domain/search";
import { CrateReferenceView } from "../features/crates/crate-reference";
import { GuidedView } from "../features/guided/guided-view";

const LazyAtlasView = lazy(async () => {
  const module = await import("../features/atlas/atlas-view");
  return { default: module.AtlasView };
});
const presentationRoutes = presentationRoutePaths;

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
  const router = useRouter();
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  });
  const storage = getAudienceStorage();
  const audience = resolveAudience(search.audience, storage);
  const routeIndex = presentationRoutes.findIndex(
    (route) => route === pathname,
  );
  const presentationAvailable = routeSupports(pathname, "presentation");
  const presentation = search.present === true && presentationAvailable;
  const filtersAvailable = routeSupports(pathname, "filters");
  const atlasStateAvailable = routeSupports(pathname, "atlasState");

  useEffect(() => {
    persistAudience(audience, storage);
    const normalizedSearch = {
      audience,
      modes: filtersAvailable ? search.modes : undefined,
      statuses: filtersAvailable ? search.statuses : undefined,
      present: presentationAvailable ? search.present : undefined,
      layout: atlasStateAvailable ? search.layout : undefined,
      ownership: atlasStateAvailable ? search.ownership : undefined,
      query: atlasStateAvailable ? search.query : undefined,
      selected: atlasStateAvailable ? search.selected : undefined,
    };
    if (
      search.audience !== normalizedSearch.audience ||
      search.modes !== normalizedSearch.modes ||
      search.statuses !== normalizedSearch.statuses ||
      search.present !== normalizedSearch.present
      || search.layout !== normalizedSearch.layout
      || search.ownership !== normalizedSearch.ownership
      || search.query !== normalizedSearch.query
      || search.selected !== normalizedSearch.selected
    ) {
      void navigate({
        replace: true,
        to: pathname,
        search: normalizedSearch,
      });
    }
  }, [
    audience,
    atlasStateAvailable,
    filtersAvailable,
    navigate,
    pathname,
    presentationAvailable,
    search,
    storage,
  ]);

  const handleAudienceChange = (nextAudience: Audience) => {
    persistAudience(nextAudience, storage);
    void navigate({
      replace: true,
      to: pathname,
      search: (previous) => ({ ...previous, audience: nextAudience }),
    });
  };

  const setPresentation = (enabled: boolean) => {
    void navigate({
      replace: true,
      to: pathname,
      search: (previous) => ({
        ...previous,
        present: enabled ? true : undefined,
      }),
    });
  };

  const navigatePresentation = (route: (typeof presentationRoutes)[number]) => {
    const retainFilters = routeSupports(route, "filters");
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries({
      audience,
      modes: retainFilters ? search.modes : undefined,
      statuses: retainFilters ? search.statuses : undefined,
      present: true,
    })) {
      if (value !== undefined) {
        params.set(key, String(value));
      }
    }
    void router.navigate({ href: `${route}?${params.toString()}` });
  };

  return (
    <AppShell
      audience={audience}
      nextRoute={presentationRoutes[routeIndex + 1]}
      onExitPresentation={() => setPresentation(false)}
      onNavigatePresentation={navigatePresentation}
      onStartPresentation={() => setPresentation(true)}
      onAudienceChange={handleAudienceChange}
      presentation={presentation}
      presentationAvailable={presentationAvailable}
      previousRoute={presentationRoutes[routeIndex - 1]}
    >
      <AudienceProvider audience={audience}>
        <Outlet />
      </AudienceProvider>
    </AppShell>
  );
}

const rootRoute = createRootRoute({
  component: RootRouteComponent,
  validateSearch: parseAtlasSearch,
});

function GuidedRouteComponent({ route }: { route: GuidedRoute }) {
  const audience = useAudience();
  const search = rootRoute.useSearch();
  const navigate = rootRoute.useNavigate();
  const updateModes = (modes: ExecutionMode[]) => {
    void navigate({
      replace: true,
      to: route,
      search: (previous) => ({
        ...previous,
        modes: encodeSelection(modes),
      }),
    });
  };
  const updateStatuses = (statuses: ArchitectureStatus[]) => {
    void navigate({
      replace: true,
      to: route,
      search: (previous) => ({
        ...previous,
        statuses: encodeSelection(statuses),
      }),
    });
  };
  return (
    <GuidedView
      audience={audience}
      modes={
        routeSupports(route, "filters") ? parseModes(search.modes) : []
      }
      onModesChange={updateModes}
      onStatusesChange={updateStatuses}
      route={route}
      statuses={
        routeSupports(route, "filters")
          ? parseStatuses(search.statuses)
          : []
      }
    />
  );
}

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => <GuidedRouteComponent route="/" />,
});

const journeyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/journey",
  component: () => <GuidedRouteComponent route="/journey" />,
});

const executionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/execution",
  component: () => <GuidedRouteComponent route="/execution" />,
});

const dataPlaneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-plane",
  component: () => <GuidedRouteComponent route="/data-plane" />,
});

const observabilityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/observability",
  component: () => <GuidedRouteComponent route="/observability" />,
});

const parityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/parity",
  component: () => <GuidedRouteComponent route="/parity" />,
});

const atlasRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/atlas",
  component: AtlasRouteComponent,
});

function AtlasRouteComponent() {
  const audience = useAudience();
  const search = rootRoute.useSearch();
  const navigate = rootRoute.useNavigate();
  return (
    <Suspense fallback={<p role="status">Loading architecture graph…</p>}>
      <LazyAtlasView
        audience={audience}
        onStateChange={(change) => {
          void navigate({
            replace: true,
            to: "/atlas",
            search: (previous) => ({
              ...previous,
              layout: change.layout ?? previous.layout,
              modes:
                change.modes === undefined
                  ? previous.modes
                  : encodeSelection(change.modes),
              ownership:
                change.owners === undefined
                  ? previous.ownership
                  : encodeSelection(change.owners),
              query:
                change.query === undefined
                  ? previous.query
                  : change.query.trim()
                    ? change.query
                    : undefined,
              selected:
                "selected" in change ? change.selected : previous.selected,
              statuses:
                change.statuses === undefined
                  ? previous.statuses
                  : encodeSelection(change.statuses),
            }),
          });
        }}
        state={{
          layout: search.layout ?? "ownership",
          modes: parseModes(search.modes),
          owners: parseOwnership(search.ownership),
          query: search.query ?? "",
          selected: search.selected,
          statuses: parseStatuses(search.statuses),
        }}
      />
    </Suspense>
  );
}

const crateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/crates/$crateId",
  component: CrateRouteComponent,
});

function CrateRouteComponent() {
  const { crateId } = crateRoute.useParams();
  return <CrateReferenceView audience={useAudience()} crateId={crateId} />;
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
