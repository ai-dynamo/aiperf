/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

import { parseDocument } from "../src/index.js";

const stdlibRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../stdlib",
);

const flagshipStubs = [
  "TokenSpanMorph.flow",
  "PromptSegmentComposer.flow",
  "RequestLifecycleWaterfall.flow",
] as const;

describe("stdlib flagship stubs", () => {
  for (const fileName of flagshipStubs) {
    test(`${fileName} parses as language 1`, () => {
      const filePath = path.join(stdlibRoot, fileName);
      const source = readFileSync(filePath, "utf8");
      const result = parseDocument(source, fileName);

      expect(result.ok, JSON.stringify(result.diagnostics, null, 2)).toBe(true);
      if (!result.ok) {
        return;
      }

      expect(result.value.language.version).toBe(1);
      expect(result.value.symbols).toHaveLength(1);
      expect(result.value.symbols[0]?.name).toBe(fileName.replace(".flow", ""));
    });
  }
});
