/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Symbol export table for `.flow` documents.

import type { DocumentAst, SymbolDefinitionAst } from "../language/index.js";
import {
  diagnostic,
  hasErrors,
  type Diagnostic,
  type Result,
} from "../schema/index.js";

/** Declared symbols keyed by export name. */
export type SymbolTable = ReadonlyMap<string, SymbolDefinitionAst>;

/** Resolves a document's symbol export table and diagnoses duplicate exports. */
export function collectSymbols(document: DocumentAst): Result<SymbolTable> {
  const table = new Map<string, SymbolDefinitionAst>();
  const diagnostics: Diagnostic[] = [];

  for (const declaration of document.symbols) {
    if (table.has(declaration.name)) {
      diagnostics.push(
        diagnostic(
          "SYMBOL_DUPLICATE_EXPORT",
          "error",
          `Duplicate symbol export "${declaration.name}".`,
          declaration.sourceMap,
          `Rename this symbol or remove the earlier "${declaration.name}" declaration.`,
        ),
      );
      continue;
    }
    table.set(declaration.name, declaration);
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return { ok: true, value: table, diagnostics };
}

export type { SymbolDefinitionAst as SymbolDeclarationAst } from "../language/index.js";
