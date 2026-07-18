// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const routeCapabilities = [
  {
    path: "/",
    label: "Runtime composition",
    sceneId: "scene.runtime-composition",
    graphState: true,
  },
  {
    path: "/scenes/runner-protocol-registries",
    label: "Runner protocol and registries",
    sceneId: "scene.runner-protocol-registries",
    graphState: true,
  },
  {
    path: "/scenes/scheduling-phase-lifecycle",
    label: "Scheduling and phase lifecycle",
    sceneId: "scene.scheduling-phase-lifecycle",
    graphState: true,
  },
  {
    path: "/scenes/dataset-segment-pipeline",
    label: "Dataset and segment pipeline",
    sceneId: "scene.dataset-segment-pipeline",
    graphState: true,
  },
  {
    path: "/scenes/endpoint-bindings-transports",
    label: "Endpoint bindings and HTTP/gRPC transports",
    sceneId: "scene.endpoint-bindings-transports",
    graphState: true,
  },
  {
    path: "/scenes/graph-ir-execution",
    label: "Graph-IR execution",
    sceneId: "scene.graph-ir-execution",
    graphState: true,
  },
  {
    path: "/scenes/metrics-telemetry",
    label: "Metrics and telemetry",
    sceneId: "scene.metrics-telemetry",
    graphState: true,
  },
  {
    path: "/scenes/accuracy-evaluator-hosting",
    label: "Accuracy and evaluator hosting",
    sceneId: "scene.accuracy-evaluator-hosting",
    graphState: true,
  },
  {
    path: "/scenes/crate-dependency-topology",
    label: "Crate dependency topology",
    sceneId: "scene.crate-dependency-topology",
    graphState: true,
  },
] as const;

export type AtlasRoutePath = (typeof routeCapabilities)[number]["path"];
export type SceneRoute = (typeof routeCapabilities)[number];
export type SceneId = SceneRoute["sceneId"];
export type RouteCapability = "graphState";

export const canonicalSceneRoutePaths: AtlasRoutePath[] = routeCapabilities.map(
  ({ path }) => path,
);
export const canonicalSceneIds: SceneId[] = routeCapabilities.map(
  ({ sceneId }) => sceneId,
);

export const legacyGuidedRedirects = {
  "/journey": "/",
  "/execution": "/scenes/endpoint-bindings-transports",
  "/data-plane": "/scenes/dataset-segment-pipeline",
  "/observability": "/scenes/metrics-telemetry",
  "/parity": "/scenes/crate-dependency-topology",
  "/atlas": "/",
} as const satisfies Record<string, AtlasRoutePath>;

export function scenePathFor(sceneId: SceneId): AtlasRoutePath {
  const route = routeCapabilities.find((candidate) => candidate.sceneId === sceneId);
  if (!route) {
    return "/";
  }
  return route.path;
}

export function sceneIdForPath(pathname: string): SceneId {
  const route = routeCapabilities.find((candidate) => candidate.path === pathname);
  return route?.sceneId ?? "scene.runtime-composition";
}

export function routeSupports(
  pathname: string,
  capability: RouteCapability,
): boolean {
  return (
    routeCapabilities.find(({ path }) => path === pathname)?.[capability] === true
  );
}
