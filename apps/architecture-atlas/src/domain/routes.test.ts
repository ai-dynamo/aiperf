// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  canonicalSceneIds,
  canonicalSceneRoutePaths,
  legacyGuidedRedirects,
  routeCapabilities,
  routeSupports,
} from "./routes";

describe("route capabilities", () => {
  it("models graph-first runtime routes and legacy guided redirects", () => {
    expect(routeCapabilities.map(({ path }) => path)).toEqual([
      "/",
      "/scenes/runner-protocol-registries",
      "/scenes/scheduling-phase-lifecycle",
      "/scenes/dataset-segment-pipeline",
      "/scenes/endpoint-bindings-transports",
      "/scenes/graph-ir-execution",
      "/scenes/metrics-telemetry",
      "/scenes/accuracy-evaluator-hosting",
      "/scenes/crate-dependency-topology",
    ]);
    expect(canonicalSceneIds).toEqual([
      "scene.runtime-composition",
      "scene.runner-protocol-registries",
      "scene.scheduling-phase-lifecycle",
      "scene.dataset-segment-pipeline",
      "scene.endpoint-bindings-transports",
      "scene.graph-ir-execution",
      "scene.metrics-telemetry",
      "scene.accuracy-evaluator-hosting",
      "scene.crate-dependency-topology",
    ]);
    expect(canonicalSceneRoutePaths).toEqual(routeCapabilities.map(({ path }) => path));
    expect(routeSupports("/", "graphState")).toBe(true);
    expect(routeSupports("/scenes/metrics-telemetry", "graphState")).toBe(true);
    expect(routeSupports("/journey", "graphState")).toBe(false);
    expect(legacyGuidedRedirects).toEqual({
      "/journey": "/",
      "/execution": "/scenes/endpoint-bindings-transports",
      "/data-plane": "/scenes/dataset-segment-pipeline",
      "/observability": "/scenes/metrics-telemetry",
      "/parity": "/scenes/crate-dependency-topology",
      "/atlas": "/",
    });
  });
});
