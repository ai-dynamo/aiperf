/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure browser `.flow` compiler pipeline.

import { parseDocument } from "../language/index.js";
import {
  safeParseFlowIr,
  type CapabilityRegistryManifest,
  type ComponentCatalog,
  type FlowIr,
  type Result,
} from "../schema/index.js";

import { expandSymbolInvocations } from "./expand-symbols.js";
import { link } from "./link.js";
import { lower } from "./lower.js";
import { collectSymbols } from "./symbols.js";
import { validate } from "./validate.js";

export const FLOW_COMPILER_VERSION = 1 as const;

/** A single request to compile `.flow` source text into Flow IR. */
export type CompileRequest = Readonly<{
  source: string;
  sourceName: string;
  capabilities: CapabilityRegistryManifest;
  strict: boolean;
  /** When provided, enables COMPONENT_UNKNOWN / prop validation. */
  components?: ComponentCatalog;
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

  const linked = link(expanded.value, {
    ...(request.components === undefined
      ? {}
      : { components: request.components }),
  });
  if (!linked.ok) {
    return linked;
  }

  const validated = validate(
    linked.value,
    request.capabilities,
    request.strict,
    request.components,
  );
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
