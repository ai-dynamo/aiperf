/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  canonicalizeModuleUri,
  resolveModuleGraph,
  type InjectedModuleSource,
  type ModuleImport,
} from "../src/module-resolution.js";

const EMPTY_SHA256 =
  "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=";

function range(source: string, offset = 0): SourceRange {
  const position = { offset, line: 1, column: offset + 1 };
  return { source, start: position, end: position };
}

function dependency(
  source: string,
  specifier: string,
  integrity?: string,
): ModuleImport {
  return integrity === undefined
    ? { specifier, range: range(source) }
    : { specifier, integrity, range: range(source) };
}

function module(
  uri: string,
  imports: readonly ModuleImport[] = [],
  content = "",
): InjectedModuleSource {
  return { uri, content, languageVersion: 1, imports };
}

describe("canonicalizeModuleUri", () => {
  test("normalizes dot segments, percent escapes, and default HTTPS ports", () => {
    expect(canonicalizeModuleUri("file:///workspace/lib/../core.flow")).toBe(
      "file:///workspace/core.flow",
    );
    expect(
      canonicalizeModuleUri("https://EXAMPLE.com:443/a/%7ecore.flow"),
    ).toBe("https://example.com/a/~core.flow");
  });
});

describe("resolveModuleGraph", () => {
  test("resolves injected local and package sources into a sorted manifest", () => {
    const result = resolveModuleGraph({
      entryUri: "file:///workspace/app/main.flow",
      sourceRoots: ["file:///workspace"],
      packageMap: {
        "@aiperf/flow-stdlib/viz": {
          uri: "file:///packages/flow-stdlib/viz.flow",
          version: "2.1.0",
        },
      },
      sources: [
        module("file:///workspace/app/../shared.flow"),
        module("file:///packages/flow-stdlib/viz.flow"),
        module("file:///workspace/app/main.flow", [
          dependency("file:///workspace/app/main.flow", "../shared.flow"),
          dependency(
            "file:///workspace/app/main.flow",
            "@aiperf/flow-stdlib/viz",
          ),
        ]),
      ],
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.modules.map(({ canonicalUri }) => canonicalUri)).toEqual([
      "file:///packages/flow-stdlib/viz.flow",
      "file:///workspace/app/main.flow",
      "file:///workspace/shared.flow",
    ]);
    expect(result.value.modules[1]?.dependencies).toEqual([
      expect.objectContaining({
        specifier: "@aiperf/flow-stdlib/viz",
        canonicalUri: "file:///packages/flow-stdlib/viz.flow",
      }),
      expect.objectContaining({
        specifier: "../shared.flow",
        canonicalUri: "file:///workspace/shared.flow",
      }),
    ]);
    expect(result.value.manifest.map(({ sourceKind }) => sourceKind)).toEqual([
      "package",
      "entry",
      "local",
    ]);
    expect(result.value.manifest[0]).toEqual(
      expect.objectContaining({ packageVersion: "2.1.0" }),
    );
  });

  test.each([
    {
      name: "unsupported specifier",
      import: dependency("file:///workspace/main.flow", "/absolute.flow"),
      code: "MODULE_INVALID_SPECIFIER",
    },
    {
      name: "source-root escape",
      import: dependency("file:///workspace/main.flow", "../outside.flow"),
      code: "MODULE_OUTSIDE_SOURCE_ROOT",
    },
    {
      name: "missing injected source",
      import: dependency("file:///workspace/main.flow", "./missing.flow"),
      code: "MODULE_NOT_FOUND",
    },
  ])("rejects $name", ({ import: imported, code }) => {
    const result = resolveModuleGraph({
      entryUri: "file:///workspace/main.flow",
      sourceRoots: ["file:///workspace"],
      sources: [module("file:///workspace/main.flow", [imported])],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({ code, severity: "error" }),
    ]);
  });

  test.each([
    {
      name: "missing integrity",
      integrity: undefined,
      code: "MODULE_INTEGRITY_REQUIRED",
    },
    {
      name: "malformed integrity",
      integrity: "md5-deadbeef",
      code: "MODULE_INTEGRITY_INVALID",
    },
    {
      name: "digest mismatch",
      integrity: "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
      code: "MODULE_INTEGRITY_MISMATCH",
    },
  ])("rejects a remote import with $name", ({ integrity, code }) => {
    const imported = dependency(
      "file:///workspace/main.flow",
      "https://cdn.example.com/lib.flow",
      integrity,
    );
    const result = resolveModuleGraph({
      entryUri: "file:///workspace/main.flow",
      sourceRoots: ["file:///workspace"],
      allowedRemoteOrigins: ["https://cdn.example.com"],
      sources: [
        module("file:///workspace/main.flow", [imported]),
        module("https://cdn.example.com/lib.flow"),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({ code, severity: "error" }),
    ]);
  });

  test("verifies injected remote bytes and records integrity metadata", () => {
    const result = resolveModuleGraph({
      entryUri: "file:///workspace/main.flow",
      sourceRoots: ["file:///workspace"],
      allowedRemoteOrigins: ["https://cdn.example.com"],
      sources: [
        module("file:///workspace/main.flow", [
          dependency(
            "file:///workspace/main.flow",
            "https://cdn.example.com/lib.flow",
            EMPTY_SHA256,
          ),
        ]),
        module("https://cdn.example.com/lib.flow"),
      ],
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.manifest[1]).toEqual(
      expect.objectContaining({
        canonicalUri: "https://cdn.example.com/lib.flow",
        resolverIdentity: `https://cdn.example.com/lib.flow#integrity=${EMPTY_SHA256}`,
        sourceKind: "remote",
        contentDigest: EMPTY_SHA256,
        remoteIntegrity: EMPTY_SHA256,
      }),
    );
  });

  test("rejects denied remote origins before consulting injected sources", () => {
    const result = resolveModuleGraph({
      entryUri: "file:///workspace/main.flow",
      sourceRoots: ["file:///workspace"],
      allowedRemoteOrigins: [],
      sources: [
        module("file:///workspace/main.flow", [
          dependency(
            "file:///workspace/main.flow",
            "https://cdn.example.com/lib.flow",
            EMPTY_SHA256,
          ),
        ]),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({ code: "MODULE_REMOTE_ORIGIN_DENIED" }),
    ]);
  });

  test("reports a deterministic shortest import cycle on its closing edge", () => {
    const result = resolveModuleGraph({
      entryUri: "file:///workspace/a.flow",
      sourceRoots: ["file:///workspace"],
      sources: [
        module("file:///workspace/c.flow", [
          dependency("file:///workspace/c.flow", "./a.flow", undefined),
        ]),
        module("file:///workspace/a.flow", [
          dependency("file:///workspace/a.flow", "./b.flow", undefined),
        ]),
        module("file:///workspace/b.flow", [
          dependency("file:///workspace/b.flow", "./c.flow", undefined),
        ]),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "MODULE_IMPORT_CYCLE",
        range: range("file:///workspace/c.flow"),
        message:
          "Module import cycle: file:///workspace/a.flow → file:///workspace/b.flow → file:///workspace/c.flow → file:///workspace/a.flow.",
        repair:
          "Extract shared declarations into a module outside this dependency cycle.",
      }),
    ]);
  });
});
