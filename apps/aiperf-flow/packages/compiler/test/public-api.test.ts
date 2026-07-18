/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import {
  compileExplainerSource,
  lowerExplainerToDeckPackage,
  packDeckPackageToJson,
  validateExplainerSet,
  writeDeckPackage,
} from "../src/index.js";

describe("@aiperf/flow-compiler public API", () => {
  test("exports explainer DeckPackage entry points", () => {
    expect(compileExplainerSource).toBeTypeOf("function");
    expect(lowerExplainerToDeckPackage).toBeTypeOf("function");
    expect(packDeckPackageToJson).toBeTypeOf("function");
    expect(writeDeckPackage).toBeTypeOf("function");
    expect(validateExplainerSet).toBeTypeOf("function");
  });
});
