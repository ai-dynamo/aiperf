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
