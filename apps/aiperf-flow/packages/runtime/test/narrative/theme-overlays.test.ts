// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

function stylesheet(name: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../../src/narrative/${name}`, import.meta.url)),
    "utf8",
  );
}

describe("narrative overlay theme contracts", () => {
  test.each(["audio-consent-modal.css", "subtitle-overlay.css"])(
    "%s uses Systems Chalk surfaces without glass effects",
    (name) => {
      const css = stylesheet(name);

      expect(css).toContain("--flow-panel");
      expect(css).toContain("--flow-chalk");
      expect(css).toContain("--flow-signal");
      expect(css).not.toContain("backdrop-filter");
      expect(css).not.toContain("box-shadow");
    },
  );
});
