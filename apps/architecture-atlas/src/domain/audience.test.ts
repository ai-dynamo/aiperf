// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  AUDIENCE_STORAGE_KEY,
  DEFAULT_AUDIENCE,
  audienceSchema,
  readStoredAudience,
  resolveAudience,
} from "./audience";

describe("audience state", () => {
  it("models the three supported audience modes", () => {
    expect(audienceSchema.options).toEqual([
      "executive",
      "developer",
      "maintainer",
    ]);
  });

  it("gives a valid URL audience precedence over persisted state", () => {
    const storage = {
      getItem: (key: string) =>
        key === AUDIENCE_STORAGE_KEY ? "maintainer" : null,
    };

    expect(resolveAudience("executive", storage)).toBe("executive");
  });

  it("uses persisted state when the URL omits an audience", () => {
    const storage = {
      getItem: () => "maintainer",
    };

    expect(resolveAudience(undefined, storage)).toBe("maintainer");
  });

  it("falls back safely when persisted state is invalid", () => {
    const storage = {
      getItem: () => "operator",
    };

    expect(readStoredAudience(storage)).toBe(DEFAULT_AUDIENCE);
  });

  it("falls back safely when storage is unavailable", () => {
    const storage = {
      getItem: () => {
        throw new DOMException("denied");
      },
    };

    expect(readStoredAudience(storage)).toBe(DEFAULT_AUDIENCE);
  });
});
