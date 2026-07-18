// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CapabilityDescriptor } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  CapabilityRegistry,
  DuplicateCapabilityError,
  type RuntimeCapability,
} from "../src/registry.js";

function capability(id: string): RuntimeCapability {
  return {
    descriptor: {
      id,
      version: "1.0.0",
      kind: "primitive",
      description: `${id} test capability`,
      nodeKinds: ["rect"],
      deterministic: true,
      accessibility: {
        requiresLabel: true,
        keyboardOperable: false,
        screenReaderFallback: true,
      },
      fallback: "core.group",
      cost: { base: 1, perNode: 1 },
    } satisfies CapabilityDescriptor,
    render: () => null,
  };
}

describe("CapabilityRegistry", () => {
  test("rejects duplicate runtime capability IDs", () => {
    const registry = new CapabilityRegistry();
    registry.register(capability("core.rect"));

    expect(() => registry.register(capability("core.rect"))).toThrow(
      DuplicateCapabilityError,
    );
  });

  test("requires only registered capabilities", () => {
    const registry = new CapabilityRegistry();
    const rect = capability("core.rect");
    registry.register(rect);

    expect(registry.require("core.rect")).toBe(rect);
    expect(() => registry.require("core.text")).toThrow(
      'Runtime capability "core.text" is not registered.',
    );
  });

  test("returns a stable manifest ordered by capability ID", () => {
    const registry = new CapabilityRegistry();
    registry.register(capability("core.text"));
    registry.register(capability("core.rect"));

    expect(
      registry.manifest().capabilities.map((descriptor) => descriptor.id),
    ).toEqual(["core.rect", "core.text"]);
  });
});
