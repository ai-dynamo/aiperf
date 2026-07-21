/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { pathEndpointsCoincident, pathPoints } from "./geometry.mjs";
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

function packageWithFan() {
  return {
    id: "fan-snapshot-contract",
    slides: [
      {
        id: "slide-1",
        title: "Fan snapshot contract",
        render: {
          scene: {
            viewport: { width: 700, height: 400 },
            roots: [
              {
                id: "fan",
                kind: "fan",
                capabilityId: "core.fan-out",
                geometry: { x: 0, y: 0, width: 0, height: 0 },
                from: { x: 0, y: 100 },
                to: [
                  { x: 300, y: 50 },
                  { x: 300, y: 150 },
                ],
              },
            ],
            timeline: [
              { id: "fan-trace", at: 0, duration: 500, action: "trace", target: "fan" },
            ],
          },
        },
      },
    ],
  };
}

function snapshotWithFan() {
  return {
    deckId: "fan-snapshot-contract",
    slideId: "slide-1",
    snapshot: {
      viewport: { width: 700, height: 400 },
      nodes: [
        {
          id: "fan",
          capability: "core.fan-out",
          bounds: { x: 0, y: 0, width: 0, height: 0 },
          ancestorIds: [],
        },
      ],
      generatedParts: [],
      connectors: [],
      fans: [
        {
          id: "fan",
          capability: "core.fan-out",
          junction: { x: 150, y: 100 },
          segments: [],
          trajectories: [
            { id: "fan-trajectory-0", d: "M0 100 L150 100 L150 50 L300 50", role: "branch" },
            { id: "fan-trajectory-1", d: "M0 100 L150 100 L150 150 L300 150", role: "branch" },
          ],
        },
      ],
      diagnostics: [],
    },
  };
}

function snapshotWithoutFans() {
  const snapshot = snapshotWithFan();
  return {
    ...snapshot,
    snapshot: { ...snapshot.snapshot, fans: [] },
  };
}

function snapshotWithDisconnectedFan() {
  const snapshot = snapshotWithFan();
  return {
    ...snapshot,
    snapshot: {
      ...snapshot.snapshot,
      fans: [
        {
          ...snapshot.snapshot.fans[0],
          // Junction no longer sits on either trajectory.
          junction: { x: 999, y: 999 },
        },
      ],
    },
  };
}

describe("flow verifier canonical fan geometry contract", () => {
  it("does not flag a fan whose canonical snapshot geometry is connected", () => {
    const findings = verifyPackageIr(packageWithFan(), {
      snapshots: [snapshotWithFan()],
    });

    expect(findings).not.toContainEqual(
      expect.objectContaining({ code: "resolved-fan-missing" }),
    );
    expect(findings).not.toContainEqual(
      expect.objectContaining({ code: "fan-disconnected-junction" }),
    );
    expect(findings).not.toContainEqual(
      expect.objectContaining({ code: "fan-invalid-cardinality" }),
    );
  });

  it("fails closed when a fan is absent from the canonical resolved snapshot", () => {
    const findings = verifyPackageIr(packageWithFan(), {
      snapshots: [snapshotWithoutFans()],
    });

    expect(findings).toContainEqual(
      expect.objectContaining({
        severity: "error",
        code: "resolved-fan-missing",
      }),
    );
  });

  it("rejects canonical fan geometry whose junction does not connect to its trajectories", () => {
    const findings = verifyPackageIr(packageWithFan(), {
      snapshots: [snapshotWithDisconnectedFan()],
    });

    expect(findings).toContainEqual(
      expect.objectContaining({
        severity: "error",
        code: "fan-disconnected-junction",
      }),
    );
  });
});

describe("flow verifier path degeneracy", () => {
  it("flags zero-length connector paths", () => {
    const pts = pathPoints("M10 20 L10 20");
    expect(pathEndpointsCoincident(pts)).toBe(true);
  });
});
