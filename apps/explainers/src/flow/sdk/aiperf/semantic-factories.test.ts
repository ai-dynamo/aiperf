/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { RenderNodeIr } from "../../schema/ir.js";
import { AIPERF_SDK_COMPONENTS } from "../registry.js";

const SOURCE_MAP = {
  source: "aiperf-semantic-factories.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

const REPRESENTATIVE_CONTRACTS: Readonly<
  Record<string, Readonly<{ ports: readonly string[]; actions: readonly string[] }>>
> = {
  "aiperf.controllerCells": {
    ports: ["controller", "cell[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.workerMerge": {
    ports: ["result", "worker[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.registryBootstrap": {
    ports: ["registry", "category[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.requestPipeline": {
    ports: ["input", "output", "stage[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.segmentPool": {
    ports: ["pool", "segment[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.warmupHandoff": {
    ports: ["from", "to"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.veloEnvelope": {
    ports: ["envelope", "payload"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.phaseLifecycle": {
    ports: ["phase[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
  "aiperf.metricsExport": {
    ports: ["metrics", "exporter[0]"],
    actions: ["enter", "draw", "trace", "emphasis"],
  },
};

function descendants(nodes: readonly RenderNodeIr[]): readonly RenderNodeIr[] {
  return nodes.flatMap((node) => [
    node,
    ...(node.kind === "group" || node.kind === "component"
      ? descendants(node.children)
      : []),
  ]);
}

describe("AIPerf semantic SDK factories", () => {
  it("emits native semantic nodes without generated chrome or label descendants", () => {
    expect(AIPERF_SDK_COMPONENTS.map((definition) => definition.descriptor.id).sort()).toEqual(
      Object.keys(REPRESENTATIVE_CONTRACTS).sort(),
    );

    for (const definition of AIPERF_SDK_COMPONENTS) {
      const instanceId = definition.descriptor.id.replace(".", "-");
      const result = definition.factory(
        { id: instanceId },
        {},
        { instanceId, sourceMap: SOURCE_MAP, themeTokens: new Map() },
      );

      expect(result.ok, definition.descriptor.id).toBe(true);
      if (!result.ok) {
        continue;
      }

      const nodes = descendants(result.value.roots);
      expect(result.value.roots.map((root) => root.id), definition.descriptor.id).toEqual([
        instanceId,
      ]);
      expect(
        nodes
          .filter((node) => /(?:-chrome|-label)$/.test(node.id))
          .map((node) => node.id),
        definition.descriptor.id,
      ).toEqual([]);
      expect(
        nodes.some(
          (node) => node.capabilityId === "core.panel" || node.capabilityId === "core.chip",
        ),
        definition.descriptor.id,
      ).toBe(true);
      expect(
        nodes.every(
          (node) =>
            node.sourceMap === SOURCE_MAP &&
            typeof node.accessibility.label === "string" &&
            node.accessibility.label.length > 0,
        ),
        definition.descriptor.id,
      ).toBe(true);
    }
  });

  it("retains representative public ports and actions", () => {
    for (const definition of AIPERF_SDK_COMPONENTS) {
      const instanceId = definition.descriptor.id.replace(".", "-");
      const result = definition.factory(
        { id: instanceId },
        {},
        { instanceId, sourceMap: SOURCE_MAP, themeTokens: new Map() },
      );
      const contract = REPRESENTATIVE_CONTRACTS[definition.descriptor.id]!;

      expect(result.ok, definition.descriptor.id).toBe(true);
      if (!result.ok) {
        continue;
      }
      expect(Object.keys(result.value.ports), definition.descriptor.id).toEqual(
        expect.arrayContaining([...contract.ports]),
      );
      expect(Object.keys(result.value.actions), definition.descriptor.id).toEqual(
        expect.arrayContaining([...contract.actions]),
      );
      const nodeIds = new Set(descendants(result.value.roots).map((node) => node.id));
      expect(
        Object.values(result.value.ports).every(
          (endpoint) => endpoint.nodeId === undefined || nodeIds.has(endpoint.nodeId),
        ),
        definition.descriptor.id,
      ).toBe(true);
      expect(
        Object.values(result.value.actions)
          .flat()
          .every((nodeId) => nodeIds.has(nodeId)),
        definition.descriptor.id,
      ).toBe(true);
    }
  });
});
