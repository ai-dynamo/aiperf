/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { createSdkRegistry } from "../registry.js";
import type { SdkExpansionContext } from "../types.js";

const SOURCE_MAP = {
  source: "deck-composites.test.flow",
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

describe("sdk.sectionDivider", () => {
  it("expands with minimal valid props and roots a core.group of text children", () => {
    const definition = createSdkRegistry().lookup("sdk.sectionDivider")!;
    const result = definition.factory(
      { id: "sd", number: "01", title: "Two Seams" },
      {},
      context("sd"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    const root = result.value.roots[0]!;
    expect(root.kind).toBe("group");
    expect(root.capabilityId).toBe("core.group");
    expect(result.value.ports.number).toEqual({ nodeId: "sd__number" });
    expect(result.value.ports.title).toEqual({ nodeId: "sd__title" });
  });

  it("emits a diagnostic (not a throw) when the required title prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.sectionDivider")!;
    const result = definition.factory({ id: "sd", number: "01" }, {}, context("sd"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.stepChain", () => {
  it("expands with minimal valid props and roots a core.group of step boxes", () => {
    const definition = createSdkRegistry().lookup("sdk.stepChain")!;
    const result = definition.factory(
      { id: "sc", steps: [{ number: "01", label: "VALIDATE" }, { number: "02", label: "SELECT" }] },
      {},
      context("sc"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("core.group");
    expect(result.value.ports["step[0]"]).toEqual({ nodeId: "sc__step-0" });
    expect(result.value.ports["step[1]"]).toEqual({ nodeId: "sc__step-1" });
    // One arrow between the two steps.
    expect(result.value.ports["arrow[0]"]).toEqual({ nodeId: "sc__arrow-0" });
    const root = result.value.roots[0];
    const arrow =
      root?.kind === "group"
        ? root.children.find((child) => child.id === "sc__arrow-0")
        : undefined;
    expect(arrow).toMatchObject({
      geometry: { x: 168, y: 58, width: 46, height: 0 },
      from: { nodeId: "sc__step-0", anchor: "e" },
      to: { nodeId: "sc__step-1", anchor: "w" },
    });
  });

  it("emits a diagnostic (not a throw) when the required steps prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.stepChain")!;
    const result = definition.factory({ id: "sc" }, {}, context("sc"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.bigStat", () => {
  it("expands with minimal valid props and roots a core.group with a value text", () => {
    const definition = createSdkRegistry().lookup("sdk.bigStat")!;
    const result = definition.factory({ id: "bs", value: "3" }, {}, context("bs"));

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("core.group");
    expect(result.value.ports.value).toEqual({ nodeId: "bs__value" });
  });

  it("emits a diagnostic (not a throw) when the required value prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.bigStat")!;
    const result = definition.factory({ id: "bs" }, {}, context("bs"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.compareGrid", () => {
  it("expands with minimal valid props and roots a layout.grid of cells", () => {
    const definition = createSdkRegistry().lookup("sdk.compareGrid")!;
    const result = definition.factory(
      { id: "cg", items: [{ label: "Clock" }, { label: "Dispatch" }, { label: "Transport" }] },
      {},
      context("cg"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("layout.grid");
    expect(result.value.ports["cell[0]"]).toEqual({ nodeId: "cg__cell-0" });
    expect(result.value.ports["cell[2]"]).toEqual({ nodeId: "cg__cell-2" });
  });

  it("emits a diagnostic (not a throw) when the required items prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.compareGrid")!;
    const result = definition.factory({ id: "cg" }, {}, context("cg"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.numberedSequence", () => {
  it("expands with minimal valid props and roots a core.group of indexed rows", () => {
    const definition = createSdkRegistry().lookup("sdk.numberedSequence")!;
    const result = definition.factory(
      {
        id: "ns",
        items: [
          { number: "1", title: "on_arrival", emphasis: true },
          { number: "2", title: "on_admit", detail: "gate cleared" },
        ],
      },
      {},
      context("ns"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("core.group");
    expect(result.value.ports["row[0]"]).toEqual({ nodeId: "ns__row-0" });
    expect(result.value.ports["row[1]"]).toEqual({ nodeId: "ns__row-1" });
  });

  it("emits a diagnostic (not a throw) when the required items prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.numberedSequence")!;
    const result = definition.factory({ id: "ns" }, {}, context("ns"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.timelineAxis", () => {
  it("expands with ticks, markers, and a dashed target into a core.group axis", () => {
    const definition = createSdkRegistry().lookup("sdk.timelineAxis")!;
    const result = definition.factory(
      {
        id: "ta",
        start: 0,
        end: 3,
        unit: "ms",
        ticks: [
          { at: 0, label: "0ms" },
          { at: 1, label: "1ms" },
        ],
        markers: [
          { at: 1, label: "timerfd exact", style: "exact" },
          { at: 2.4, label: "wheel late", style: "late" },
        ],
        target: { at: 1, label: "target" },
      },
      {},
      context("ta"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("core.group");
    expect(result.value.ports["tick[0]"]).toEqual({ nodeId: "ta__tick-0" });
    expect(result.value.ports["marker[1]"]).toEqual({ nodeId: "ta__marker-1" });
    expect(result.value.ports.target).toEqual({ nodeId: "ta__target" });
  });

  it("emits a diagnostic (not a throw) when the required end prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.timelineAxis")!;
    const result = definition.factory({ id: "ta", start: 0 }, {}, context("ta"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.nodeTree", () => {
  it("paints an explicit backdrop for an emphasized root", () => {
    const definition = createSdkRegistry().lookup("sdk.nodeTree")!;
    const result = definition.factory(
      {
        id: "nt",
        root: { label: "pop", emphasis: true },
        children: [{ label: "t=1ms" }],
      },
      {},
      context("nt"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const tree = result.value.roots[0];
    const rootBox =
      tree?.kind === "group"
        ? tree.children.find((child) => child.id === "nt__root")
        : undefined;
    expect(rootBox?.kind).toBe("group");
    expect(rootBox?.kind === "group" ? rootBox.children[0] : undefined).toMatchObject({
      id: "nt__root__backdrop",
      capabilityId: "core.rect",
      style: { fill: "#76B900" },
    });
  });

  it("expands with a root, children, and caption into a core.group tree", () => {
    const definition = createSdkRegistry().lookup("sdk.nodeTree")!;
    const result = definition.factory(
      {
        id: "nt",
        root: { label: "(100, 0)", emphasis: true },
        children: [{ label: "(140, 1)" }, { label: "(140, 2)" }],
        orderNote: "pop order → (100,0) → (140,1) → (140,2)",
      },
      {},
      context("nt"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("core.group");
    expect(result.value.ports.rootBox).toEqual({ nodeId: "nt__root" });
    expect(result.value.ports["child[0]"]).toEqual({ nodeId: "nt__child-0" });
    expect(result.value.ports["child[1]"]).toEqual({ nodeId: "nt__child-1" });
    expect(result.value.ports.caption).toEqual({ nodeId: "nt__caption" });
    const tree = result.value.roots[0];
    const firstLine =
      tree?.kind === "group"
        ? tree.children.find((child) => child.id === "nt__line-0")
        : undefined;
    expect(firstLine).toMatchObject({
      geometry: { x: 75, y: 60, width: 95, height: 70 },
      from: { nodeId: "nt__root", anchor: "s" },
      to: { nodeId: "nt__child-0", anchor: "n" },
    });
  });

  it("emits a diagnostic (not a throw) when the required children prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.nodeTree")!;
    const result = definition.factory(
      { id: "nt", root: { label: "root" } },
      {},
      context("nt"),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});

describe("sdk.cardGrid", () => {
  it("expands with minimal valid props and roots a layout.grid of cards", () => {
    const definition = createSdkRegistry().lookup("sdk.cardGrid")!;
    const result = definition.factory(
      {
        id: "cg2",
        cards: [
          { title: "loadgen-core", detail: "pure engine", accent: "green" },
          { title: "aiperf", detail: "cli + engine", accent: "black" },
        ],
      },
      {},
      context("cg2"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value.roots[0]!.capabilityId).toBe("layout.grid");
    expect(result.value.ports["card[0]"]).toEqual({ nodeId: "cg2__card-0" });
    expect(result.value.ports["card[1]"]).toEqual({ nodeId: "cg2__card-1" });
  });

  it("emits a diagnostic (not a throw) when the required cards prop is absent", () => {
    const definition = createSdkRegistry().lookup("sdk.cardGrid")!;
    const result = definition.factory({ id: "cg2" }, {}, context("cg2"));

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("SDK_PROP_REQUIRED");
  });
});
