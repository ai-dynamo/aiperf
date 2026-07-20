/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { createSdkRegistry } from "../registry.js";
import type { SceneFragment, SdkExpansionContext } from "../types.js";
import type { RenderNodeIr } from "../../schema/ir.js";

const SOURCE_MAP = {
  source: "composites.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function context(instanceId: string): SdkExpansionContext {
  return {
    instanceId,
    sourceMap: SOURCE_MAP,
    themeTokens: new Map(),
  };
}

function rectNode(
  id: string,
  geometry: { x: number; y: number; width: number; height: number } = {
    x: 0,
    y: 0,
    width: 80,
    height: 40,
  },
): RenderNodeIr {
  return {
    kind: "rect",
    id,
    capabilityId: "core.rect",
    geometry,
    style: {},
    accessibility: { label: id },
    fallback: id,
    sourceMap: SOURCE_MAP,
  };
}

/** A fragment with a resolvable root id, so `portOrRootEndpoint` succeeds via fallback. */
function rootedFragment(id: string): SceneFragment {
  return {
    roots: [rectNode(id)],
    ports: {},
    actions: { enter: [id] },
  };
}

/** A fragment with no roots and no ports: `portOrRootEndpoint` cannot resolve any endpoint. */
const ROOTLESS_FRAGMENT: SceneFragment = { roots: [], ports: {}, actions: {} };

describe("sdk.hubSpoke fail-closed endpoint resolution", () => {
  it("fails closed instead of inventing a {x:0,y:0} origin when the hub fragment has no root or port", () => {
    const definition = createSdkRegistry().lookup("sdk.hubSpoke")!;
    const result = definition.factory(
      { id: "hs" },
      { hub: [ROOTLESS_FRAGMENT], spokes: [rootedFragment("spoke-0")] },
      context("hs"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_HUB_SPOKE_HUB_ENDPOINT_UNRESOLVED");
  });

  it("fails closed when a spoke fragment has no root or port", () => {
    const definition = createSdkRegistry().lookup("sdk.hubSpoke")!;
    const result = definition.factory(
      { id: "hs" },
      { hub: [rootedFragment("hub-0")], spokes: [rootedFragment("spoke-0"), ROOTLESS_FRAGMENT] },
      context("hs"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_HUB_SPOKE_SPOKE_ENDPOINT_UNRESOLVED");
  });

  it("wires hub-to-spoke edges when every fragment resolves an endpoint", () => {
    const definition = createSdkRegistry().lookup("sdk.hubSpoke")!;
    const result = definition.factory(
      { id: "hs" },
      { hub: [rootedFragment("hub-0")], spokes: [rootedFragment("spoke-0"), rootedFragment("spoke-1")] },
      context("hs"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.ports.hub).toEqual({ nodeId: "hub-0" });
    expect(result.value.ports["spoke[0]"]).toEqual({ nodeId: "spoke-0" });
    expect(result.value.ports["spoke[1]"]).toEqual({ nodeId: "spoke-1" });
    const edgeIds = result.value.roots
      .filter((node) => node.kind === "connector")
      .map((node) => node.id);
    expect(edgeIds).toHaveLength(2);
  });
});

describe("sdk.tree fail-closed endpoint resolution", () => {
  it("fails closed instead of inventing a {x:0,y:0} origin when the root fragment has no root or port", () => {
    const definition = createSdkRegistry().lookup("sdk.tree")!;
    const result = definition.factory(
      { id: "tr" },
      { root: [ROOTLESS_FRAGMENT], children: [rootedFragment("child-0")] },
      context("tr"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_TREE_ROOT_ENDPOINT_UNRESOLVED");
  });

  it("fails closed when a child fragment has no root or port", () => {
    const definition = createSdkRegistry().lookup("sdk.tree")!;
    const result = definition.factory(
      { id: "tr" },
      { root: [rootedFragment("root-0")], children: [ROOTLESS_FRAGMENT] },
      context("tr"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_TREE_CHILD_ENDPOINT_UNRESOLVED");
  });

  it("wires root-to-child edges when every fragment resolves an endpoint", () => {
    const definition = createSdkRegistry().lookup("sdk.tree")!;
    const result = definition.factory(
      { id: "tr" },
      { root: [rootedFragment("root-0")], children: [rootedFragment("child-0")] },
      context("tr"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.ports.root).toEqual({ nodeId: "root-0" });
    expect(result.value.ports["child[0]"]).toEqual({ nodeId: "child-0" });
  });
});

describe("sdk.stateTransition fail-closed endpoint resolution", () => {
  it("fails closed instead of inventing a {x:0,y:0} origin when the from fragment has no root or port", () => {
    const definition = createSdkRegistry().lookup("sdk.stateTransition")!;
    const result = definition.factory(
      { id: "st" },
      { from: [ROOTLESS_FRAGMENT], to: [rootedFragment("to-0")] },
      context("st"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_STATE_TRANSITION_FROM_ENDPOINT_UNRESOLVED");
  });

  it("fails closed when the to fragment has no root or port", () => {
    const definition = createSdkRegistry().lookup("sdk.stateTransition")!;
    const result = definition.factory(
      { id: "st" },
      { from: [rootedFragment("from-0")], to: [ROOTLESS_FRAGMENT] },
      context("st"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_STATE_TRANSITION_TO_ENDPOINT_UNRESOLVED");
  });

  it("wires a from-to transition edge when both fragments resolve an endpoint", () => {
    const definition = createSdkRegistry().lookup("sdk.stateTransition")!;
    const result = definition.factory(
      { id: "st" },
      { from: [rootedFragment("from-0")], to: [rootedFragment("to-0")] },
      context("st"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.ports.from).toEqual({ nodeId: "from-0" });
    expect(result.value.ports.to).toEqual({ nodeId: "to-0" });
  });
});
