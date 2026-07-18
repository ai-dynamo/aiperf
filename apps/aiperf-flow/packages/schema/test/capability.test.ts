/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import {
  FOUNDATION_CAPABILITIES,
  createCapabilityManifest,
  type CapabilityDescriptor,
} from "../src/index.js";

const descriptor = (id: string): CapabilityDescriptor => ({
  id,
  version: "1.0.0",
  kind: "primitive",
  description: `${id} capability`,
  nodeKinds: ["rect"],
  deterministic: true,
  accessibility: {
    requiresLabel: true,
    keyboardOperable: false,
    screenReaderFallback: true,
  },
  fallback: "core.rect",
  cost: {
    base: 1,
    perNode: 1,
  },
});

describe("createCapabilityManifest", () => {
  it("sorts descriptors by capability ID", () => {
    const result = createCapabilityManifest([
      descriptor("core.text"),
      descriptor("core.rect"),
    ]);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.capabilities.map(({ id }) => id)).toEqual([
        "core.rect",
        "core.text",
      ]);
    }
  });

  it("rejects duplicate capability IDs", () => {
    const result = createCapabilityManifest([
      descriptor("core.rect"),
      descriptor("core.rect"),
    ]);

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "CAPABILITY_DUPLICATE",
        severity: "error",
      }),
    ]);
  });
});

describe("foundation capabilities", () => {
  it("registers the required versioned capabilities", () => {
    expect(
      FOUNDATION_CAPABILITIES.capabilities.map(({ id, version }) => `${id}@${version}`),
    ).toEqual([
      "core.camera@1.0.0",
      "core.connector@1.0.0",
      "core.group@1.0.0",
      "core.inspect@1.0.0",
      "core.rect@1.0.0",
      "core.text@1.0.0",
      "core.timeline@1.0.0",
    ]);
  });
});
