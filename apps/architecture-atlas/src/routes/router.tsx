// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
  useRouterState,
} from "@tanstack/react-router";
import type { RouterHistory } from "@tanstack/react-router";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { z } from "zod";

import { AudienceProvider, useAudience } from "../app/audience-context";
import { AppShell } from "../app/app-shell";
import { architectureCatalog } from "../content";
import {
  persistAudience,
  resolveAudience,
  audienceSchema,
} from "../domain/audience";
import {
  executionFlavorSchema,
} from "../domain/architecture";
import {
  canonicalGraphState,
  decodeGraphStateFromUrl,
  encodeGraphStateForUrl,
  resetManualLayoutState,
  resolveGraphState,
  writeStoredGraphState,
  type CanonicalGraphStateDomain,
  type GraphStateStorage,
} from "../domain/graph-state";
import {
  canonicalSceneIds,
  legacyGuidedRedirects,
  routeCapabilities,
  scenePathFor,
  type AtlasRoutePath,
  type SceneId,
} from "../domain/routes";
import { CrateReferenceView } from "../features/crates/crate-reference";
import {
  GraphScene,
} from "../features/graph/graph-scene";
import { RuntimeStory } from "../features/story/runtime-story";
import type { GraphFitViewCommand } from "../features/graph/types";

const GRAPH_SEARCH_INPUT_ID = "atlas-graph-search";
interface GraphFitLifecycle {
  command: GraphFitViewCommand | null;
  complete(requestId: number): void;
}

const GraphFitRequestContext = createContext<GraphFitLifecycle>({
  command: null,
  complete: () => undefined,
});

const unavailableStorage: GraphStateStorage = {
  getItem: () => null,
  removeItem: () => undefined,
  setItem: () => undefined,
};

const searchSchema = z.object({
  audience: audienceSchema.optional(),
  primary: executionFlavorSchema.optional(),
  compare: executionFlavorSchema.optional(),
  q: z.string().max(160).optional(),
  s: z.string().optional(),
});

type RouterSearch = z.infer<typeof searchSchema>;

function parseRouterSearch(search: Record<string, unknown>): RouterSearch {
  const parsed: RouterSearch = {};
  for (const [key, result] of Object.entries({
    audience: audienceSchema.optional().safeParse(search.audience),
    primary: executionFlavorSchema.optional().safeParse(search.primary),
    compare: executionFlavorSchema.optional().safeParse(search.compare),
    q: z.string().max(160).optional().safeParse(search.q),
    s: z.string().optional().safeParse(search.s),
  })) {
    if (result.success && result.data !== undefined) {
      Object.assign(parsed, { [key]: result.data });
    }
  }
  return parsed;
}

function getStorage(): GraphStateStorage {
  try {
    return window.localStorage;
  } catch {
    return unavailableStorage;
  }
}

function buildCanonicalDomain(defaultState: ReturnType<typeof canonicalGraphState>): CanonicalGraphStateDomain {
  return {
    defaultState,
    sceneIds: new Set(canonicalSceneIds),
    nodeIds: new Set(architectureCatalog.graphNodes.map(({ id }) => id)),
    edgeIds: new Set(architectureCatalog.graphEdges.map(({ id }) => id)),
    supportedFlavors: new Set(executionFlavorSchema.options),
  };
}

function RootRouteComponent() {
  const navigate = rootRoute.useNavigate();
  const location = useRouterState({
    select: (state) => state.location,
  });
  const pathname = location.pathname;
  const search = parseRouterSearch(location.search as Record<string, unknown>);
  const storage = getStorage();
  const activeSceneRoute = routeCapabilities.find((route) => route.path === pathname);
  const isSceneRoutePath = activeSceneRoute !== undefined;
  const isStoryRoute = pathname === "/story";

  const audience = resolveAudience(search.audience, storage);
  const routeSceneId = activeSceneRoute?.sceneId ?? "scene.runtime-composition";
  const primaryFlavor = search.primary ?? "native_http";
  const compareFlavor = search.compare ?? null;
  const defaultGraphState = canonicalGraphState({
    sceneId: routeSceneId,
    audience,
    primaryFlavor,
    compareFlavor,
  });
  const resolvedGraphState = resolveGraphState({
    urlState: search.s ?? null,
    storage,
    canonical: buildCanonicalDomain(defaultGraphState),
  });
  const [sharedStateNotice, setSharedStateNotice] = useState(
    resolvedGraphState.notice?.message,
  );
  const [fitViewCommand, setFitViewCommand] =
    useState<GraphFitViewCommand | null>(null);
  const fitRequestSequence = useRef(0);
  const completeFitView = useCallback((requestId: number) => {
    setFitViewCommand((current) =>
      current?.requestId === requestId ? null : current,
    );
  }, []);
  const fitLifecycle = useMemo(
    () => ({ command: fitViewCommand, complete: completeFitView }),
    [completeFitView, fitViewCommand],
  );
  const effectiveGraphState = canonicalGraphState({
    ...resolvedGraphState.state,
    audience,
    sceneId: routeSceneId,
    primaryFlavor,
    compareFlavor,
  });
  const encodedState = encodeGraphStateForUrl(effectiveGraphState);
  const resetEncodedState = encodeGraphStateForUrl(
    resetManualLayoutState(
      effectiveGraphState,
    ),
  );
  const legacyRedirect = legacyGuidedRedirects[pathname as keyof typeof legacyGuidedRedirects];

  useEffect(() => {
    if (!legacyRedirect) {
      return;
    }
    void navigate({
      replace: true,
      to: legacyRedirect,
      search: (previous) => ({
        ...previous,
        audience,
      }),
    });
  }, [audience, legacyRedirect, navigate]);

  useEffect(() => {
    if (resolvedGraphState.notice) {
      setSharedStateNotice(resolvedGraphState.notice.message);
    }
  }, [resolvedGraphState.notice]);

  useEffect(() => {
    persistAudience(audience, storage);
    if (!isSceneRoutePath) {
      return;
    }
    writeStoredGraphState(storage, effectiveGraphState);
    if (
      search.audience !== audience ||
      search.primary !== primaryFlavor ||
      search.compare !== (compareFlavor ?? undefined) ||
      search.s !== encodedState
    ) {
      void navigate({
        replace: true,
        to: pathname as AtlasRoutePath,
        search: (previous) => ({
          ...previous,
          audience,
          primary: primaryFlavor,
          compare: compareFlavor ?? undefined,
          s: encodedState,
        }),
      });
    }
  }, [
    audience,
    compareFlavor,
    effectiveGraphState,
    encodedState,
    navigate,
    pathname,
    primaryFlavor,
    search.audience,
    search.compare,
    search.primary,
    search.s,
    storage,
    isSceneRoutePath,
  ]);

  return (
    <AppShell
      activeScenePath={scenePathFor(routeSceneId)}
      audience={audience}
      compareFlavor={compareFlavor}
      graphSearchInputId={GRAPH_SEARCH_INPUT_ID}
      graphSearchQuery={search.q ?? ""}
      onAudienceChange={(nextAudience) => {
        void navigate({
          replace: true,
          to: pathname as AtlasRoutePath,
          search: (previous) => ({ ...previous, audience: nextAudience }),
        });
      }}
      onCompareFlavorChange={(nextCompareFlavor) => {
        void navigate({
          replace: true,
          to: pathname as AtlasRoutePath,
          search: (previous) => ({
            ...previous,
            compare: nextCompareFlavor ?? undefined,
          }),
        });
      }}
      onFitGraph={() => {
        fitRequestSequence.current += 1;
        setFitViewCommand({ requestId: fitRequestSequence.current });
      }}
      onGraphSearchChange={(query) => {
        void navigate({
          replace: true,
          to: pathname as AtlasRoutePath,
          search: (previous) => ({
            ...previous,
            q: query.trim() ? query : undefined,
          }),
        });
      }}
      onPrimaryFlavorChange={(nextPrimaryFlavor) => {
        void navigate({
          replace: true,
          to: pathname as AtlasRoutePath,
          search: (previous) => ({
            ...previous,
            primary: nextPrimaryFlavor,
            compare:
              previous.compare === nextPrimaryFlavor
                ? undefined
                : previous.compare,
          }),
        });
      }}
      onResetGraph={() => {
        if (!isSceneRoutePath) {
          return;
        }
        void navigate({
          replace: true,
          to: pathname as AtlasRoutePath,
          search: (previous) => ({ ...previous, s: resetEncodedState }),
        });
      }}
      onShareGraphState={() => {
        if (!isSceneRoutePath) {
          return;
        }
        const shareUrl = new URL(window.location.href);
        shareUrl.searchParams.set("audience", audience);
        shareUrl.searchParams.set("primary", primaryFlavor);
        if (compareFlavor) {
          shareUrl.searchParams.set("compare", compareFlavor);
        } else {
          shareUrl.searchParams.delete("compare");
        }
        if (search.q) {
          shareUrl.searchParams.set("q", search.q);
        } else {
          shareUrl.searchParams.delete("q");
        }
        shareUrl.searchParams.set("s", encodedState);
        void navigator.clipboard?.writeText(shareUrl.toString()).catch(() => undefined);
        void navigate({
          replace: true,
          to: pathname as AtlasRoutePath,
          search: (previous) => ({ ...previous, s: encodedState }),
        });
      }}
      primaryFlavor={primaryFlavor}
      sceneRoutes={routeCapabilities}
      sharedStateNotice={sharedStateNotice}
      storyMode={isStoryRoute}
    >
      <AudienceProvider audience={audience}>
        <GraphFitRequestContext.Provider value={fitLifecycle}>
          <Outlet />
        </GraphFitRequestContext.Provider>
      </AudienceProvider>
    </AppShell>
  );
}

const rootRoute = createRootRoute({
  component: RootRouteComponent,
  validateSearch: parseRouterSearch,
});

function GraphSceneRouteComponent({ sceneId }: { sceneId: SceneId }) {
  const audience = useAudience();
  const fitLifecycle = useContext(GraphFitRequestContext);
  const navigate = rootRoute.useNavigate();
  const locationSearch = useRouterState({
    select: (routerState) => routerState.location.search,
  });
  const search = parseRouterSearch(locationSearch as Record<string, unknown>);
  const primaryFlavor = search.primary ?? "native_http";
  const compareFlavor = search.compare ?? null;
  const defaultState = canonicalGraphState({
    audience,
    compareFlavor,
    primaryFlavor,
    sceneId,
  });
  const sharedState = search.s
    ? decodeGraphStateFromUrl(search.s, buildCanonicalDomain(defaultState)).state
    : defaultState;
  const state = canonicalGraphState({
    ...sharedState,
    audience,
    compareFlavor,
    primaryFlavor,
    sceneId,
  });

  return (
    <GraphScene
      audience={audience}
      compareFlavor={compareFlavor}
      fallbackFocusElementId={GRAPH_SEARCH_INPUT_ID}
      fitViewCommand={fitLifecycle.command}
      onFitViewComplete={fitLifecycle.complete}
      onGraphStateChange={(nextState) => {
        void navigate({
          replace: true,
          to: scenePathFor(sceneId),
          search: (previous) => ({
            ...previous,
            s: encodeGraphStateForUrl(
              canonicalGraphState({
                ...nextState,
                audience,
                compareFlavor,
                primaryFlavor,
                sceneId,
              }),
            ),
          }),
        });
      }}
      primaryFlavor={primaryFlavor}
      sceneId={sceneId}
      searchQuery={search.q ?? ""}
      state={state}
    />
  );
}

const runtimeSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => <GraphSceneRouteComponent sceneId="scene.runtime-composition" />,
});

const storyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/story",
  component: () => <RuntimeStory audience={useAudience()} />,
});

const runnerProtocolSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/runner-protocol-registries",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.runner-protocol-registries" />
  ),
});

const schedulingSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/scheduling-phase-lifecycle",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.scheduling-phase-lifecycle" />
  ),
});

const datasetSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/dataset-segment-pipeline",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.dataset-segment-pipeline" />
  ),
});

const endpointSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/endpoint-bindings-transports",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.endpoint-bindings-transports" />
  ),
});

const graphIrSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/graph-ir-execution",
  component: () => <GraphSceneRouteComponent sceneId="scene.graph-ir-execution" />,
});

const metricsSceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/metrics-telemetry",
  component: () => <GraphSceneRouteComponent sceneId="scene.metrics-telemetry" />,
});

const accuracySceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/accuracy-evaluator-hosting",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.accuracy-evaluator-hosting" />
  ),
});

const crateTopologySceneRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scenes/crate-dependency-topology",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.crate-dependency-topology" />
  ),
});

const journeyRedirectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/journey",
  component: () => <GraphSceneRouteComponent sceneId="scene.runtime-composition" />,
});

const executionRedirectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/execution",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.endpoint-bindings-transports" />
  ),
});

const dataPlaneRedirectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-plane",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.dataset-segment-pipeline" />
  ),
});

const observabilityRedirectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/observability",
  component: () => <GraphSceneRouteComponent sceneId="scene.metrics-telemetry" />,
});

const parityRedirectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/parity",
  component: () => (
    <GraphSceneRouteComponent sceneId="scene.crate-dependency-topology" />
  ),
});

const atlasRedirectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/atlas",
  component: () => <GraphSceneRouteComponent sceneId="scene.runtime-composition" />,
});

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
  runtimeSceneRoute,
  storyRoute,
  runnerProtocolSceneRoute,
  schedulingSceneRoute,
  datasetSceneRoute,
  endpointSceneRoute,
  graphIrSceneRoute,
  metricsSceneRoute,
  accuracySceneRoute,
  crateTopologySceneRoute,
  journeyRedirectRoute,
  executionRedirectRoute,
  dataPlaneRedirectRoute,
  observabilityRedirectRoute,
  parityRedirectRoute,
  atlasRedirectRoute,
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
