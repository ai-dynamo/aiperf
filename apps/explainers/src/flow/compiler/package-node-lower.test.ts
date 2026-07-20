/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

const compilerModules = import.meta.glob("./*.ts", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

describe("package node lowering module", () => {
  it("contains only the live package-node lowering surface", () => {
    expect(compilerModules).not.toHaveProperty("./desugar-scene-primitives.ts");

    const source = compilerModules["./package-node-lower.ts"];
    expect(source).toBeDefined();
    expect(source).toContain("export function asRecord");
    expect(source).toContain("export function capabilityKind");
    expect(source).toContain("export function isSupportedPackageCapability");
    expect(source).toContain("export function lowerFirstClassPackageNode");
    expect(source).not.toContain("desugarPackageNode");
  });
});
