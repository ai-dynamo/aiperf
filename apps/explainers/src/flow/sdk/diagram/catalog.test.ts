/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, render } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { SceneRenderer } from "../../../core/diagram/SceneRenderer.js";
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

afterEach(cleanup);

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

    expect(storage.ok && storage.value.ports.write).toEqual({
      nodeId: "node",
      anchor: "w",
    });
    expect(storage.ok && storage.value.ports.read).toEqual({
      nodeId: "node",
      anchor: "e",
    });
    expect(messaging.ok && messaging.value.ports.producer).toBeDefined();
    expect(messaging.ok && messaging.value.ports.consumer).toBeDefined();
    expect(network.ok && network.value.ports.inbound).toBeDefined();
    expect(network.ok && network.value.ports.outbound).toBeDefined();

    const retry = registry.lookup("sdk.retry")!.factory({ id: "node" }, {}, context);
    expect(retry.ok && retry.value.ports.back).toEqual({
      nodeId: "node",
      anchor: "s",
    });
    expect(retry.ok && retry.value.ports.back && "x" in retry.value.ports.back).toBe(false);
    expect(retry.ok && retry.value.ports.back && "y" in retry.value.ports.back).toBe(false);
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

      const resolved = resolveScene({
        roots: result.value.roots,
        timeline: [],
      });
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

  it.each(["sdk.retry", "sdk.loop"] as const)(
    "%s decorative back-edge is a non-routing path attached at non-zero world coords",
    (componentId) => {
      const registry = createSdkRegistry();
      const result = registry.lookup(componentId)!.factory(
        {
          id: "hero",
          title: "Reconnect",
          x: 729,
          y: 337.5,
          width: 459,
          height: 270,
        },
        {},
        {
          instanceId: "hero",
          sourceMap: SOURCE_MAP,
          themeTokens: new Map(),
        },
      );

      expect(result.ok).toBe(true);
      if (!result.ok) {
        return;
      }

      const backNode = (() => {
        const root = result.value.roots[0];
        if (root?.kind !== "group") {
          return undefined;
        }
        return root.children.find((child) => child.id === "hero__back-edge");
      })();
      expect(backNode).toMatchObject({
        id: "hero__back-edge",
        capabilityId: "core.path",
      });
      // Schema-required local dummies only — never nodeId routing anchors.
      expect(backNode).toMatchObject({
        from: { x: expect.any(Number), y: expect.any(Number) },
        to: { x: expect.any(Number), y: expect.any(Number) },
      });
      expect(
        backNode && "from" in backNode && backNode.from && "nodeId" in backNode.from
          ? backNode.from.nodeId
          : undefined,
      ).toBeUndefined();
      expect(
        backNode && "to" in backNode && backNode.to && "nodeId" in backNode.to
          ? backNode.to.nodeId
          : undefined,
      ).toBeUndefined();
      expect(
        typeof backNode?.path === "string" ? backNode.path.length : 0,
      ).toBeGreaterThan(0);

      const resolved = resolveScene({
        roots: result.value.roots,
        timeline: [],
      });
      const nodeBounds = resolved.worldGeometryById.get("hero");
      const loopBounds = resolved.worldGeometryById.get("hero__back-edge");

      expect(nodeBounds).toMatchObject({ x: 729, y: 337.5, width: 459, height: 270 });
      // Decorative loop must not enter connector routing / geometry diagnostics.
      expect(resolved.connectorsById.has("hero__back-edge")).toBe(false);
      expect(
        resolved.diagnostics.filter((diagnostic) =>
          diagnostic.nodeIds.includes("hero__back-edge"),
        ),
      ).toEqual([]);

      expect(loopBounds).toBeDefined();
      if (nodeBounds === undefined || loopBounds === undefined) {
        return;
      }

      // Loop geometry sits on the bottom edge of the Retry/Loop node.
      expect(loopBounds.x).toBeGreaterThan(200);
      expect(loopBounds.y + tipInsetSafe(loopBounds)).toBeGreaterThanOrEqual(
        nodeBounds.y + nodeBounds.height - 2,
      );
      expect(loopBounds.y).toBeLessThanOrEqual(nodeBounds.y + nodeBounds.height);
      expect(loopBounds.x + loopBounds.width).toBeLessThanOrEqual(
        nodeBounds.x + nodeBounds.width + 4,
      );

      expect(result.value.actions.draw).toContain("hero__back-edge");
      expect(result.value.actions.trace).toContain("hero__back-edge");

      function tipInsetSafe(
        bounds: Readonly<{ height: number }>,
      ): number {
        // Path origin is tipInset above the bottom edge (see catalog factory).
        return Math.min(21.6, Math.max(0, bounds.height - 48.6));
      }
    },
  );

  it.each(["sdk.retry", "sdk.loop"] as const)(
    "%s back-edge paints a bottom loop in SceneRenderer",
    (componentId) => {
      const result = createSdkRegistry().lookup(componentId)!.factory(
        {
          id: "hero",
          title: "Reconnect",
          x: 729,
          y: 337.5,
          width: 459,
          height: 270,
          strokeRole: "@theme.accent.green",
        },
        {},
        {
          instanceId: "hero",
          sourceMap: SOURCE_MAP,
          themeTokens: new Map(),
        },
      );
      expect(result.ok).toBe(true);
      if (!result.ok) {
        return;
      }

      const { container } = render(
        createElement(SceneRenderer, {
          scene: { id: "retry-loop", roots: result.value.roots, timeline: [] },
          playing: false,
          restartKey: 0,
        }),
      );
      const node = container.querySelector('[data-flow-node-id="hero__back-edge"]');
      const path = node?.querySelector("path");
      expect(path).not.toBeNull();
      const d =
        path?.getAttribute("data-flow-resolved-path") ?? path?.getAttribute("d") ?? "";
      expect(d).toMatch(/^M/);
      // Local glyph path is translated by the loop's world origin under the hero.
      const translated = [...(node?.querySelectorAll("g[transform]") ?? [])].some(
        (group) => {
          const transform = group.getAttribute("transform") ?? "";
          return /translate\(\s*77[0-9](?:\.\d+)?[\s,]+58[0-9](?:\.\d+)?\s*\)/.test(
            transform,
          );
        },
      );
      const worldPath = /M\s*11[0-9]\d(?:\.\d+)?[\s,]+\s*6[0-9]\d(?:\.\d+)?/.test(d);
      expect(translated || worldPath).toBe(true);
    },
  );
});
