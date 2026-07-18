/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic module-graph resolution over explicitly injected Flow sources.
//!
//! This seam canonicalizes module identities, applies local/package/remote
//! policy, verifies remote integrity, and diagnoses dependency cycles. It
//! deliberately accepts the complete source set as data and performs no
//! filesystem or network access.

import {
  diagnostic,
  type Diagnostic,
  type Result,
  type SourceRange,
} from "@aiperf/flow-schema";
import { sha256 } from "js-sha256";

/** An authored import edge extracted from one injected module. */
export type ModuleImport = Readonly<{
  specifier: string;
  range: SourceRange;
  integrity?: string;
}>;

/** One source module supplied by the resolver host. */
export type InjectedModuleSource = Readonly<{
  uri: string;
  content: string | Uint8Array;
  languageVersion: number;
  imports: readonly ModuleImport[];
}>;

/** Exact immutable package-map entry supplied by the build host. */
export type PackageResolution = Readonly<{
  uri: string;
  version: string;
}>;

/** Complete input to deterministic module resolution. */
export type ModuleResolutionRequest = Readonly<{
  entryUri: string;
  sources: readonly InjectedModuleSource[];
  sourceRoots: readonly string[];
  packageMap?: Readonly<Record<string, PackageResolution>>;
  allowedRemoteOrigins?: readonly string[];
}>;

export type ModuleSourceKind = "entry" | "local" | "package" | "remote";

/** A resolved import edge retaining its authored location. */
export type ResolvedModuleDependency = Readonly<{
  specifier: string;
  canonicalUri: string;
  range: SourceRange;
  integrity?: string;
}>;

/** One reachable source and its canonical direct dependencies. */
export type ResolvedModule = Readonly<{
  canonicalUri: string;
  resolverIdentity: string;
  sourceKind: ModuleSourceKind;
  contentDigest: string;
  languageVersion: number;
  content: string | Uint8Array;
  dependencies: readonly ResolvedModuleDependency[];
  packageVersion?: string;
  remoteIntegrity?: string;
}>;

/** Portable, deterministic metadata for one reachable module. */
export type ModuleManifestEntry = Readonly<{
  canonicalUri: string;
  resolverIdentity: string;
  sourceKind: ModuleSourceKind;
  contentDigest: string;
  languageVersion: number;
  dependencies: readonly string[];
  packageVersion?: string;
  remoteIntegrity?: string;
}>;

/** Fully resolved graph and its sorted dependency manifest. */
export type ResolvedModuleGraph = Readonly<{
  entryUri: string;
  modules: readonly ResolvedModule[];
  manifest: readonly ModuleManifestEntry[];
}>;

type ResolutionTarget = Readonly<{
  canonicalUri: string;
  sourceKind: Exclude<ModuleSourceKind, "entry">;
  packageVersion?: string;
  integrity?: string;
}>;

const BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const SHA256_INTEGRITY = /^sha256-[A-Za-z0-9+/]{43}=$/;
const UNRESERVED_ESCAPE = /%([0-9A-Fa-f]{2})/g;

function normalizeEscapes(value: string): string {
  return value.replace(UNRESERVED_ESCAPE, (escape, hex: string) => {
    const character = String.fromCharCode(Number.parseInt(hex, 16));
    return /^[A-Za-z0-9\-._~]$/.test(character)
      ? character
      : escape.toUpperCase();
  });
}

/** Returns the stable URI identity used to key injected sources and graph nodes. */
export function canonicalizeModuleUri(uri: string): string {
  const parsed = new URL(uri);
  parsed.hash = "";
  if (parsed.protocol === "file:") {
    parsed.search = "";
  }
  return normalizeEscapes(parsed.href);
}

function bytes(content: string | Uint8Array): Uint8Array {
  return typeof content === "string" ? new TextEncoder().encode(content) : content;
}

function hexToBase64(hex: string): string {
  let encoded = "";
  for (let index = 0; index < hex.length; index += 6) {
    const chunk = hex.slice(index, index + 6);
    const byteCount = chunk.length / 2;
    const value = Number.parseInt(chunk.padEnd(6, "0"), 16);
    encoded += BASE64[(value >>> 18) & 63] ?? "";
    encoded += BASE64[(value >>> 12) & 63] ?? "";
    encoded += byteCount > 1 ? (BASE64[(value >>> 6) & 63] ?? "") : "=";
    encoded += byteCount > 2 ? (BASE64[value & 63] ?? "") : "=";
  }
  return encoded;
}

function contentDigest(content: string | Uint8Array): string {
  return `sha256-${hexToBase64(sha256(bytes(content)))}`;
}

function syntheticRange(source: string): SourceRange {
  const position = { offset: 0, line: 1, column: 1 };
  return { source, start: position, end: position };
}

function isWithinRoot(uri: string, roots: readonly string[]): boolean {
  return roots.some((root) => {
    const prefix = root.endsWith("/") ? root : `${root}/`;
    return uri === root || uri.startsWith(prefix);
  });
}

function isExactFlowModule(uri: string): boolean {
  const parsed = new URL(uri);
  return parsed.pathname.endsWith(".flow") && parsed.search === "" && parsed.hash === "";
}

function invalidSpecifier(imported: ModuleImport): Diagnostic {
  return diagnostic(
    "MODULE_INVALID_SPECIFIER",
    "error",
    `Invalid module specifier "${imported.specifier}".`,
    imported.range,
    "Use an exact relative .flow path, configured package name, or HTTPS URL.",
  );
}

function resolveImport(
  importerUri: string,
  imported: ModuleImport,
  roots: readonly string[],
  packageMap: Readonly<Record<string, PackageResolution>>,
  allowedOrigins: ReadonlySet<string>,
): Result<ResolutionTarget> {
  const { specifier } = imported;
  if (specifier.startsWith("./") || specifier.startsWith("../")) {
    const canonicalUri = canonicalizeModuleUri(new URL(specifier, importerUri).href);
    if (!isExactFlowModule(canonicalUri)) {
      return { ok: false, diagnostics: [invalidSpecifier(imported)] };
    }
    if (!isWithinRoot(canonicalUri, roots)) {
      return {
        ok: false,
        diagnostics: [
          diagnostic(
            "MODULE_OUTSIDE_SOURCE_ROOT",
            "error",
            `Local import "${specifier}" resolves outside every configured source root.`,
            imported.range,
            "Move the module under a configured source root or change the import.",
          ),
        ],
      };
    }
    return {
      ok: true,
      value: { canonicalUri, sourceKind: "local" },
      diagnostics: [],
    };
  }

  const packageResolution = packageMap[specifier];
  if (packageResolution !== undefined) {
    const canonicalUri = canonicalizeModuleUri(packageResolution.uri);
    if (!isExactFlowModule(canonicalUri)) {
      return { ok: false, diagnostics: [invalidSpecifier(imported)] };
    }
    return {
      ok: true,
      value: {
        canonicalUri,
        sourceKind: "package",
        packageVersion: packageResolution.version,
      },
      diagnostics: [],
    };
  }

  if (specifier.startsWith("https://")) {
    let canonicalUri: string;
    try {
      canonicalUri = canonicalizeModuleUri(specifier);
    } catch {
      return { ok: false, diagnostics: [invalidSpecifier(imported)] };
    }
    if (!isExactFlowModule(canonicalUri)) {
      return { ok: false, diagnostics: [invalidSpecifier(imported)] };
    }
    if (!allowedOrigins.has(new URL(canonicalUri).origin)) {
      return {
        ok: false,
        diagnostics: [
          diagnostic(
            "MODULE_REMOTE_ORIGIN_DENIED",
            "error",
            `Remote module origin "${new URL(canonicalUri).origin}" is not allowlisted.`,
            imported.range,
            "Use an allowlisted HTTPS origin or a local/package dependency.",
          ),
        ],
      };
    }
    if (imported.integrity === undefined) {
      return {
        ok: false,
        diagnostics: [
          diagnostic(
            "MODULE_INTEGRITY_REQUIRED",
            "error",
            `Remote import "${specifier}" requires SHA-256 integrity metadata.`,
            imported.range,
            "Add a verified sha256 Subresource Integrity value.",
          ),
        ],
      };
    }
    if (!SHA256_INTEGRITY.test(imported.integrity)) {
      return {
        ok: false,
        diagnostics: [
          diagnostic(
            "MODULE_INTEGRITY_INVALID",
            "error",
            `Remote import "${specifier}" has malformed or unsupported integrity metadata.`,
            imported.range,
            "Use a sha256- prefixed base64 digest of the exact module bytes.",
          ),
        ],
      };
    }
    return {
      ok: true,
      value: {
        canonicalUri,
        sourceKind: "remote",
        integrity: imported.integrity,
      },
      diagnostics: [],
    };
  }

  return { ok: false, diagnostics: [invalidSpecifier(imported)] };
}

function sortedUnique(values: readonly string[]): readonly string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

function shortestPath(
  start: string,
  target: string,
  adjacency: ReadonlyMap<string, readonly string[]>,
): readonly string[] | undefined {
  const queue: (readonly string[])[] = [[start]];
  const visited = new Set([start]);
  while (queue.length > 0) {
    const path = queue.shift();
    if (path === undefined) {
      break;
    }
    const last = path.at(-1);
    if (last === target) {
      return path;
    }
    for (const next of adjacency.get(last ?? "") ?? []) {
      if (!visited.has(next)) {
        visited.add(next);
        queue.push([...path, next]);
      }
    }
  }
  return undefined;
}

function normalizeCycle(cycle: readonly string[]): readonly string[] {
  const body = cycle.slice(0, -1);
  const rotations = body.map((_, index) => [
    ...body.slice(index),
    ...body.slice(0, index),
  ]);
  rotations.sort((left, right) => left.join("\0").localeCompare(right.join("\0")));
  const normalized = rotations[0] ?? [];
  return [...normalized, normalized[0] ?? ""];
}

function cycleDiagnostics(modules: readonly ResolvedModule[]): readonly Diagnostic[] {
  const adjacency = new Map(
    modules.map((module) => [
      module.canonicalUri,
      sortedUnique(module.dependencies.map(({ canonicalUri }) => canonicalUri)),
    ]),
  );
  const candidates = new Map<string, readonly string[]>();
  for (const [from, targets] of adjacency) {
    for (const to of targets) {
      const returnPath = shortestPath(to, from, adjacency);
      if (returnPath === undefined) {
        continue;
      }
      const cycle = normalizeCycle([...returnPath, to]);
      candidates.set(cycle.join("\0"), cycle);
    }
  }

  const cycles = [...candidates.values()].sort(
    (left, right) =>
      left.length - right.length ||
      left.join("\0").localeCompare(right.join("\0")),
  );
  const selectedComponents = new Set<string>();
  const diagnostics: Diagnostic[] = [];
  const uris = [...adjacency.keys()].sort((left, right) => left.localeCompare(right));
  for (const cycle of cycles) {
    const first = cycle[0];
    if (first === undefined) {
      continue;
    }
    const component = uris.filter(
      (uri) =>
        shortestPath(first, uri, adjacency) !== undefined &&
        shortestPath(uri, first, adjacency) !== undefined,
    );
    const componentKey = component[0] ?? first;
    if (selectedComponents.has(componentKey)) {
      continue;
    }
    selectedComponents.add(componentKey);

    const closingFrom = cycle.at(-2);
    const closingTo = cycle[0];
    const closingModule = modules.find(
      ({ canonicalUri }) => canonicalUri === closingFrom,
    );
    const closingEdge = closingModule?.dependencies.find(
      ({ canonicalUri }) => canonicalUri === closingTo,
    );
    diagnostics.push(
      diagnostic(
        "MODULE_IMPORT_CYCLE",
        "error",
        `Module import cycle: ${cycle.join(" → ")}.`,
        closingEdge?.range ?? syntheticRange(closingFrom ?? first),
        "Extract shared declarations into a module outside this dependency cycle.",
      ),
    );
  }
  return diagnostics;
}

/** Resolves the complete reachable graph using only the supplied source values. */
export function resolveModuleGraph(
  request: ModuleResolutionRequest,
): Result<ResolvedModuleGraph> {
  let entryUri: string;
  let roots: readonly string[];
  try {
    entryUri = canonicalizeModuleUri(request.entryUri);
    roots = request.sourceRoots.map(canonicalizeModuleUri).sort();
  } catch {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "MODULE_INVALID_SPECIFIER",
          "error",
          `Invalid entry module URI "${request.entryUri}".`,
          syntheticRange(request.entryUri),
        ),
      ],
    };
  }

  const sources = new Map<string, InjectedModuleSource>();
  for (const source of request.sources) {
    try {
      const canonicalUri = canonicalizeModuleUri(source.uri);
      if (!sources.has(canonicalUri)) {
        sources.set(canonicalUri, source);
      }
    } catch {
      return {
        ok: false,
        diagnostics: [
          diagnostic(
            "MODULE_INVALID_SPECIFIER",
            "error",
            `Injected source URI "${source.uri}" is invalid.`,
            syntheticRange(source.uri),
          ),
        ],
      };
    }
  }

  const entry = sources.get(entryUri);
  if (entry === undefined) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "MODULE_NOT_FOUND",
          "error",
          `Entry module "${entryUri}" was not supplied.`,
          syntheticRange(entryUri),
        ),
      ],
    };
  }

  const packageMap = request.packageMap ?? {};
  const allowedOrigins = new Set(
    (request.allowedRemoteOrigins ?? []).map((origin) => new URL(origin).origin),
  );
  const pending = [entryUri];
  const visited = new Set<string>();
  const moduleKinds = new Map<string, ResolutionTarget>([
    [entryUri, { canonicalUri: entryUri, sourceKind: "local" }],
  ]);
  const modules: ResolvedModule[] = [];
  const diagnostics: Diagnostic[] = [];

  while (pending.length > 0) {
    pending.sort((left, right) => left.localeCompare(right));
    const canonicalUri = pending.shift();
    if (canonicalUri === undefined || visited.has(canonicalUri)) {
      continue;
    }
    visited.add(canonicalUri);
    const source = sources.get(canonicalUri);
    if (source === undefined) {
      continue;
    }

    const dependencies: ResolvedModuleDependency[] = [];
    for (const imported of source.imports) {
      const resolved = resolveImport(
        canonicalUri,
        imported,
        roots,
        packageMap,
        allowedOrigins,
      );
      if (!resolved.ok) {
        diagnostics.push(...resolved.diagnostics);
        continue;
      }
      const targetSource = sources.get(resolved.value.canonicalUri);
      if (targetSource === undefined) {
        diagnostics.push(
          diagnostic(
            "MODULE_NOT_FOUND",
            "error",
            `Module "${resolved.value.canonicalUri}" was not supplied.`,
            imported.range,
            "Inject the exact resolved module source into this compilation.",
          ),
        );
        continue;
      }
      if (
        resolved.value.integrity !== undefined &&
        contentDigest(targetSource.content) !== resolved.value.integrity
      ) {
        diagnostics.push(
          diagnostic(
            "MODULE_INTEGRITY_MISMATCH",
            "error",
            `Injected bytes for "${resolved.value.canonicalUri}" do not match the import integrity digest.`,
            imported.range,
            "Supply the pinned bytes or update the integrity value through dependency refresh.",
          ),
        );
        continue;
      }

      const existing = moduleKinds.get(resolved.value.canonicalUri);
      if (existing === undefined || existing.sourceKind === "local") {
        moduleKinds.set(resolved.value.canonicalUri, resolved.value);
      }
      dependencies.push(
        resolved.value.integrity === undefined
          ? {
              specifier: imported.specifier,
              canonicalUri: resolved.value.canonicalUri,
              range: imported.range,
            }
          : {
              specifier: imported.specifier,
              canonicalUri: resolved.value.canonicalUri,
              range: imported.range,
              integrity: resolved.value.integrity,
            },
      );
      pending.push(resolved.value.canonicalUri);
    }

    const identity = moduleKinds.get(canonicalUri);
    const digest = contentDigest(source.content);
    const sourceKind: ModuleSourceKind =
      canonicalUri === entryUri ? "entry" : identity?.sourceKind ?? "local";
    const remoteIntegrity =
      sourceKind === "remote" ? identity?.integrity : undefined;
    const resolverIdentity =
      remoteIntegrity === undefined
        ? canonicalUri
        : `${canonicalUri}#integrity=${remoteIntegrity}`;
    const base = {
      canonicalUri,
      resolverIdentity,
      sourceKind,
      contentDigest: digest,
      languageVersion: source.languageVersion,
      content: source.content,
      dependencies: dependencies.sort((left, right) =>
        left.canonicalUri.localeCompare(right.canonicalUri) ||
        left.specifier.localeCompare(right.specifier),
      ),
    };
    modules.push(
      remoteIntegrity !== undefined
        ? { ...base, remoteIntegrity }
        : identity?.packageVersion !== undefined
          ? { ...base, packageVersion: identity.packageVersion }
          : base,
    );
  }

  modules.sort((left, right) => left.canonicalUri.localeCompare(right.canonicalUri));
  diagnostics.push(...cycleDiagnostics(modules));
  if (diagnostics.length > 0) {
    return { ok: false, diagnostics };
  }

  const manifest = modules.map((module): ModuleManifestEntry => {
    const base = {
      canonicalUri: module.canonicalUri,
      resolverIdentity: module.resolverIdentity,
      sourceKind: module.sourceKind,
      contentDigest: module.contentDigest,
      languageVersion: module.languageVersion,
      dependencies: sortedUnique(
        module.dependencies.map(({ canonicalUri }) => canonicalUri),
      ),
    };
    return module.remoteIntegrity !== undefined
      ? { ...base, remoteIntegrity: module.remoteIntegrity }
      : module.packageVersion !== undefined
        ? { ...base, packageVersion: module.packageVersion }
        : base;
  });
  return {
    ok: true,
    value: { entryUri, modules, manifest },
    diagnostics: [],
  };
}
