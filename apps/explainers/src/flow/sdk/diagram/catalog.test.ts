/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { resolveScene } from "../../../core/diagram/resolution/resolve-scene.js";
import { createSdkRegistry } from "../registry.js";
import type { SceneFragment } from "../types.js";

const DIAGRAM_CATALOG_IDS = [
  "sdk.user",
  "sdk.client",
  "sdk.service",
  "sdk.server",
  "sdk.process",
  "sdk.worker",
  "sdk.function",
  "sdk.container",
  "sdk.cloud",
  "sdk.database",
  "sdk.dataStore",
  "sdk.cache",
  "sdk.file",
  "sdk.objectStore",
  "sdk.volume",
  "sdk.queue",
  "sdk.topic",
  "sdk.stream",
  "sdk.eventBus",
  "sdk.gateway",
  "sdk.endpoint",
  "sdk.loadBalancer",
  "sdk.firewall",
  "sdk.start",
  "sdk.end",
  "sdk.processStep",
  "sdk.decision",
  "sdk.merge",
  "sdk.delay",
  "sdk.retry",
  "sdk.loop",
  "sdk.boundary",
  "sdk.zone",
  "sdk.cluster",
  "sdk.trustBoundary",
  "sdk.document",
  "sdk.terminal",
  "sdk.clock",
  "sdk.lock",
  "sdk.key",
  "sdk.warning",
] as const;

const SOURCE_MAP = {
  source: "diagram-catalog.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

const CHILD: SceneFragment = {
  roots: [
    {
      kind: "rect",
      id: "child",
      capabilityId: "core.rect",
      geometry: { x: 0, y: 0, width: 80, height: 40 },
      style: {},
      accessibility: { label: "child" },
      fallback: "child",
      sourceMap: SOURCE_MAP,
    },
  ],
  ports: { self: { nodeId: "child" } },
  actions: { enter: ["child"] },
};

describe("diagram SDK catalog", () => {
  it("registers every approved systems-diagram primitive", () => {
    const registry = createSdkRegistry();

    expect(DIAGRAM_CATALOG_IDS.filter((id) => registry.lookup(id) === undefined)).toEqual([]);
  });

  it("expands every diagram primitive with semantic connection ports", () => {
    const registry = createSdkRegistry();
    const failures: string[] = [];

    for (const componentId of DIAGRAM_CATALOG_IDS) {
      const definition = registry.lookup(componentId)!;
      const result = definition.factory(
        { id: "example", title: "Example", label: "Example", branches: ["yes", "no"] },
        { children: [CHILD] },
        {
          instanceId: componentId.replace(".", "-"),
          sourceMap: SOURCE_MAP,
          themeTokens: new Map(),
        },
      );
      if (!result.ok || result.value.roots.length === 0 || result.value.ports.self === undefined) {
        failures.push(componentId);
      }
    }

    expect(failures).toEqual([]);
  });

  it("publishes category-specific props and semantic ports", () => {
    const registry = createSdkRegistry();
    expect(registry.lookup("sdk.server")!.descriptor.props.branches).toBeUndefined();
    expect(registry.lookup("sdk.decision")!.descriptor.props.branches).toBeDefined();

    const context = {
      instanceId: "node",
      sourceMap: SOURCE_MAP,
      themeTokens: new Map(),
    };
    const storage = registry.lookup("sdk.database")!.factory({ id: "node" }, {}, context);
    const messaging = registry.lookup("sdk.queue")!.factory({ id: "node" }, {}, context);
    const network = registry.lookup("sdk.gateway")!.factory({ id: "node" }, {}, context);

    expect(storage.ok && storage.value.ports.read).toBeDefined();
    expect(storage.ok && storage.value.ports.write).toBeDefined();
    expect(messaging.ok && messaging.value.ports.producer).toBeDefined();
    expect(messaging.ok && messaging.value.ports.consumer).toBeDefined();
    expect(network.ok && network.value.ports.inbound).toBeDefined();
    expect(network.ok && network.value.ports.outbound).toBeDefined();
  });

  it("emits standard diagram chrome as semantic root props", () => {
    const result = createSdkRegistry().lookup("sdk.server")!.factory(
      { id: "server", title: "API", detail: "healthy" },
      {},
      {
        instanceId: "server",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      expect(root).toMatchObject({
        id: "server",
        kind: "group",
        capabilityId: "diagram.compute",
        props: { title: "API", detail: "healthy" },
        sdkOrigin: {
          componentId: "sdk.server",
          instanceId: "server",
          generatedRole: "root",
        },
      });
      expect(
        root?.kind === "group"
          ? root.children.filter(
              (child) =>
                ["chrome", "title", "detail"].includes(
                  child.sdkOrigin?.generatedRole ?? "",
                ) &&
                (child.capabilityId === "core.rect" ||
                  child.capabilityId === "core.text"),
            )
          : [],
      ).toEqual([]);
      expect(result.value.ports.icon).toEqual({ nodeId: "server__glyph" });
      expect(result.value.ports.title).toEqual({ nodeId: "server__title" });
      const descendantIds = new Set<string>();
      const collectDescendantIds = (nodes: readonly (typeof result.value.roots)[number][]) => {
        for (const node of nodes) {
          descendantIds.add(node.id);
          if (node.kind === "group" || node.kind === "component") {
            collectDescendantIds(node.children);
          }
        }
      };
      collectDescendantIds(result.value.roots);
      expect(descendantIds.has("server__title")).toBe(false);

      const resolved = resolveScene({ roots: result.value.roots });
      expect(resolved.generatedPartsById.get("server__title")).toMatchObject({
        ownerId: "server",
        role: "title",
      });
      expect(
        resolved.diagnostics.some(
          (diagnostic) => diagnostic.code === "SCENE_DUPLICATE_PAINT_OWNER",
        ),
      ).toBe(false);
      expect(result.value.actions).toEqual({
        enter: ["server"],
        emphasis: ["server"],
        exit: ["server"],
      });
    }

    const gateway = createSdkRegistry().lookup("sdk.gateway")!.factory(
      { id: "gateway", title: "Ingress" },
      {},
      {
        instanceId: "gateway",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      },
    );
    expect(gateway.ok).toBe(true);
    if (gateway.ok) {
      const emittedIds = new Set<string>();
      const collectIds = (nodes: typeof gateway.value.roots) => {
        for (const node of nodes) {
          emittedIds.add(node.id);
          if (node.kind === "group") {
            collectIds(node.children);
          }
        }
      };
      collectIds(gateway.value.roots);
      expect(gateway.value.actions).toEqual({
        enter: ["gateway"],
        draw: ["gateway"],
        trace: ["gateway"],
      });
      for (const drawId of gateway.value.actions?.draw ?? []) {
        expect(emittedIds.has(drawId)).toBe(true);
      }
    }
  });

  it("emits boundary chrome semantically while preserving authored children", () => {
    const result = createSdkRegistry().lookup("sdk.zone")!.factory(
      { id: "zone", title: "Control plane" },
      { children: [CHILD] },
      {
        instanceId: "zone",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      expect(root).toMatchObject({
        id: "zone",
        kind: "group",
        capabilityId: "diagram.boundary",
        props: { title: "Control plane" },
      });
      expect(
        root?.kind === "group"
          ? root.children.filter(
              (child) =>
                ["chrome", "title"].includes(
                  child.sdkOrigin?.generatedRole ?? "",
                ) &&
                (child.capabilityId === "core.rect" ||
                  child.capabilityId === "core.text"),
            )
          : [],
      ).toEqual([]);
      expect(result.value.ports["child[0]"]).toEqual({ nodeId: "child" });
      expect(result.value.actions).toEqual({
        enter: ["zone"],
        stagger: ["child"],
      });
    }
  });

  it("keeps emitted action bindings within each public action contract", () => {
    const registry = createSdkRegistry();
    const mismatches: string[] = [];
    for (const componentId of DIAGRAM_CATALOG_IDS) {
      const definition = registry.lookup(componentId)!;
      const result = definition.factory(
        { id: "node", branches: ["yes", "no"] },
        { children: [CHILD] },
        {
          instanceId: componentId.replace(".", "-"),
          sourceMap: SOURCE_MAP,
          themeTokens: new Map(),
        },
      );
      if (
        result.ok &&
        Object.keys(result.value.actions).some(
          (action) => !definition.actions.includes(action as never),
        )
      ) {
        mismatches.push(componentId);
      }
    }
    expect(mismatches).toEqual([]);
  });
});
