// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const routeCapabilities = [
  {
    path: "/",
    label: "System ownership",
    guided: true,
    filters: true,
    presentation: true,
    atlasState: false,
  },
  {
    path: "/journey",
    label: "One-run journey",
    guided: true,
    filters: false,
    presentation: true,
    atlasState: false,
  },
  {
    path: "/execution",
    label: "Execution modes",
    guided: true,
    filters: true,
    presentation: true,
    atlasState: false,
  },
  {
    path: "/data-plane",
    label: "Data plane",
    guided: true,
    filters: true,
    presentation: true,
    atlasState: false,
  },
  {
    path: "/observability",
    label: "Observability",
    guided: true,
    filters: true,
    presentation: true,
    atlasState: false,
  },
  {
    path: "/parity",
    label: "Parity ledger",
    guided: true,
    filters: true,
    presentation: true,
    atlasState: false,
  },
  {
    path: "/atlas",
    label: "Unified atlas",
    guided: false,
    filters: true,
    presentation: false,
    atlasState: true,
  },
] as const;

export type AtlasRoutePath = (typeof routeCapabilities)[number]["path"];
export type GuidedRoute = Extract<
  (typeof routeCapabilities)[number],
  { guided: true }
>["path"];
export type PresentationRoute = Extract<
  (typeof routeCapabilities)[number],
  { presentation: true }
>["path"];
export type RouteCapability =
  | "filters"
  | "presentation"
  | "atlasState"
  | "guided";

export const guidedRoutePaths: GuidedRoute[] = routeCapabilities
  .filter((route): route is Extract<typeof route, { guided: true }> => route.guided)
  .map(({ path }) => path);
export const presentationRoutePaths: PresentationRoute[] = routeCapabilities
  .filter(
    (route): route is Extract<typeof route, { presentation: true }> =>
      route.presentation,
  )
  .map(({ path }) => path);

export function routeSupports(
  pathname: string,
  capability: RouteCapability,
): boolean {
  return (
    routeCapabilities.find(({ path }) => path === pathname)?.[capability] === true
  );
}
