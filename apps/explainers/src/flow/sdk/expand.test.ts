/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { Diagnostic } from "../schema/index.js";
import { resolveRef, type SdkInstanceEntry, type SdkInstanceIndex } from "./expand.js";

const SOURCE_RANGE = {
  source: "expand.test.ts",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function entry(
  instanceId: string,
  ports: Readonly<Record<string, { nodeId: string }>>,
): SdkInstanceEntry {
  return {
    instanceId,
    componentId: "sdk.note",
    ports,
    actions: {},
    rootIds: [instanceId],
    sourceMap: SOURCE_RANGE,
  };
}

describe("resolveRef dotted instance ids", () => {
  it("resolves refs when the instance id itself contains dots", () => {
    const index: SdkInstanceIndex = new Map([
      ["aiperf", entry("aiperf", { output: { nodeId: "wrong" } })],
      [
        "aiperf.controller",
        entry("aiperf.controller", { output: { nodeId: "controller-out" } }),
      ],
    ]);
    const diagnostics: Diagnostic[] = [];

    const resolved = resolveRef(
      "aiperf.controller.output",
      index,
      SOURCE_RANGE,
      diagnostics,
    );

    expect(diagnostics).toEqual([]);
    expect(resolved).toEqual({ nodeId: "controller-out" });
  });

  it("still resolves indexed port families on undotted instance ids", () => {
    const index: SdkInstanceIndex = new Map([
      ["cells", entry("cells", { "worker[0]": { nodeId: "worker-0" } })],
    ]);
    const diagnostics: Diagnostic[] = [];

    const resolved = resolveRef(
      "cells.worker.0",
      index,
      SOURCE_RANGE,
      diagnostics,
    );

    expect(diagnostics).toEqual([]);
    expect(resolved).toEqual({ nodeId: "worker-0" });
  });
});
