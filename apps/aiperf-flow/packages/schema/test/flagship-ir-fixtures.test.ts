/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

import { safeParseFlowIr } from "../src/ir.js";

const examplesRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../examples/p0",
);

const flagshipFixtures = [
  "token-span-morph.ir.json",
  "prompt-segment-composer.ir.json",
  "request-lifecycle-waterfall.ir.json",
] as const;

describe("flagship IR fixtures", () => {
  for (const fixture of flagshipFixtures) {
    test(`${fixture} passes strict Flow IR validation`, () => {
      const filePath = path.join(examplesRoot, fixture);
      const input = JSON.parse(readFileSync(filePath, "utf8"));
      const parsed = safeParseFlowIr(input);

      expect(parsed.ok, JSON.stringify(parsed.diagnostics, null, 2)).toBe(true);
      if (parsed.ok) {
        expect(parsed.value.irVersion).toBe(1);
      }
    });
  }
});
