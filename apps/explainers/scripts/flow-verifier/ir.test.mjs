/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { verifyPackageIr } from "./ir.mjs";

function packageWithConnector() {
  return {
    id: "snapshot-contract",
    slides: [
      {
        id: "slide-1",
        title: "Snapshot contract",
        render: {
          scene: {
            viewport: { width: 700, height: 400 },
            roots: [
              {
                id: "source",
                kind: "rect",
                geometry: { x: 10, y: 20, width: 80, height: 40 },
              },
              {
                id: "target",
                kind: "rect",
                geometry: { x: 210, y: 20, width: 80, height: 40 },
              },
              {
                id: "edge",
                kind: "connector",
                from: { nodeId: "source", anchor: "e" },
                to: { nodeId: "target", anchor: "w" },
              },
            ],
            timeline: [{ id: "show-source", at: 0, action: "show", target: "source" }],
          },
        },
      },
    ],
  };
}

function snapshotWithoutConnectors() {
  return {
    deckId: "snapshot-contract",
    slideId: "slide-1",
    snapshot: {
      viewport: { width: 700, height: 400 },
      nodes: [
        {
          id: "source",
          capability: "core.rect",
          bounds: { x: 10, y: 20, width: 80, height: 40 },
          ancestorIds: [],
        },
        {
          id: "target",
          capability: "core.rect",
          bounds: { x: 210, y: 20, width: 80, height: 40 },
          ancestorIds: [],
        },
      ],
      generatedParts: [],
      connectors: [],
      diagnostics: [],
    },
  };
}

describe("flow verifier canonical snapshot contract", () => {
  it("rejects a scene that has no canonical resolved snapshot", () => {
    const findings = verifyPackageIr(packageWithConnector());

    expect(findings).toContainEqual(
      expect.objectContaining({
        severity: "error",
        code: "resolved-snapshot-missing",
      }),
    );
  });

  it("does not reconstruct a connector omitted from the canonical snapshot", () => {
    const findings = verifyPackageIr(packageWithConnector(), {
      snapshots: [snapshotWithoutConnectors()],
    });

    expect(findings).toContainEqual(
      expect.objectContaining({
        severity: "error",
        code: "resolved-connector-missing",
      }),
    );
  });
});
