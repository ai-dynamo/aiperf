/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `.flow` compiler pipeline: parse → symbols → link → validate → lower → pack.
//!
//! `compileSource` is the single entry point from source text to validated
//! Flow IR. Each stage returns a `Result`, and the pipeline short-circuits on
//! the first stage that fails so callers see the earliest, most actionable
//! diagnostics.

import { parseDocument } from "@aiperf/flow-language";
import {
  safeParseFlowIr,
  type CapabilityRegistryManifest,
  type FlowIr,
  type Result,
} from "@aiperf/flow-schema";

import { expandSymbolInvocations } from "./expand-symbols.js";
import { link } from "./link.js";
import { lower } from "./lower.js";
import { collectSymbols } from "./symbols.js";
import { validate } from "./validate.js";

export * from "./components.js";
export * from "./expand-symbols.js";
export * from "./link.js";
export * from "./pack.js";
export * from "./symbols.js";
export * from "./validate.js";
export { lower } from "./lower.js";

export const FLOW_COMPILER_VERSION = 1 as const;

/** A single request to compile `.flow` source text into Flow IR. */
export type CompileRequest = Readonly<{
  source: string;
  sourceName: string;
  capabilities: CapabilityRegistryManifest;
  strict: boolean;
}>;

/** Runs the full parse → symbols → link → validate → lower → schema-validate pipeline. */
export function compileSource(request: CompileRequest): Result<FlowIr> {
  const parsed = parseDocument(request.source, request.sourceName);
  if (!parsed.ok) {
    return parsed;
  }

  const symbols = collectSymbols(parsed.value);
  if (!symbols.ok) {
    return symbols;
  }

  const expanded = expandSymbolInvocations(parsed.value, symbols.value);
  if (!expanded.ok) {
    return expanded;
  }

  const linked = link(expanded.value);
  if (!linked.ok) {
    return linked;
  }

  const validated = validate(linked.value, request.capabilities, request.strict);
  if (!validated.ok) {
    return validated;
  }

  const irResult = safeParseFlowIr(lower(validated.value));
  if (!irResult.ok) {
    return irResult;
  }

  return {
    ok: true,
    value: irResult.value,
    diagnostics: [
      ...parsed.diagnostics,
      ...expanded.diagnostics,
      ...validated.diagnostics,
      ...irResult.diagnostics,
    ],
  };
}
