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
import { useEffect } from "react";

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
  encodeSelection,
  parseAtlasSearch,
  parseModes,
  parseStatuses,
} from "../domain/search";
import {
  GuidedView,
  type GuidedRoute,
} from "../features/guided/guided-view";
import { RoutePlaceholder } from "./placeholders";

const presentationRoutes = [
  "/",
  "/journey",
  "/execution",
  "/data-plane",
  "/observability",
  "/parity",
] as const;

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
  const presentation = search.present === true && routeIndex >= 0;

  useEffect(() => {
    persistAudience(audience, storage);
    if (search.audience !== audience) {
      void navigate({
        replace: true,
        to: pathname,
        search: (previous) => ({ ...previous, audience }),
      });
    }
  }, [audience, navigate, search.audience, storage]);

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
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries({
      ...search,
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
      modes={parseModes(search.modes)}
      onModesChange={updateModes}
      onStatusesChange={updateStatuses}
      route={route}
      statuses={parseStatuses(search.statuses)}
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
