// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import type { EvidenceReference } from "../../domain/architecture";
import { repositorySource } from "./source-url";

const rangedEvidence: EvidenceReference = {
  path: "crates/runner/src/grpc_execution.rs",
  lines: { start: 164, end: 195 },
};

describe("repository source URLs", () => {
  it("builds a repository URL with line fragments", () => {
    expect(
      repositorySource(
        rangedEvidence,
        "https://github.com/ai-dynamo/aiperf/blob/main",
      ),
    ).toEqual({
      label: "crates/runner/src/grpc_execution.rs:164-195",
      href: "https://github.com/ai-dynamo/aiperf/blob/main/crates/runner/src/grpc_execution.rs#L164-L195",
    });
  });

  it("returns a safe plain-path fallback without a web base", () => {
    expect(repositorySource(rangedEvidence, undefined)).toEqual({
      label: "crates/runner/src/grpc_execution.rs:164-195",
    });
  });
});
