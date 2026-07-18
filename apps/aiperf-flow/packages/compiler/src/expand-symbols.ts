/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Symbol invocation expansion.
//!
//! Local symbol calls are replaced by cloned body invocations after strict,
//! named parameter binding. Expansion is recursive and preserves authored
//! source provenance.

import type {
  ArgumentValueAst,
  ComponentInvocationAst,
  DocumentAst,
  IdentifierReferenceAst,
  RenderDeclarationAst,
  SceneAst,
  SymbolDefinitionAst,
  ValueAst,
} from "@aiperf/flow-language";
import {
  diagnostic,
  hasErrors,
  type Diagnostic,
  type JsonValue,
  type Result,
} from "@aiperf/flow-schema";

import {
  validateProps,
  type ComponentPropsSchema,
  type PropValueKind,
} from "./components.js";
import type { SymbolTable } from "./symbols.js";

type Bindings = ReadonlyMap<string, ArgumentValueAst>;

const BUILTIN_PARAM_KINDS: Readonly<Record<string, PropValueKind>> = {
  string: "string",
  number: "number",
  boolean: "boolean",
  EntityId: "string",
};

function isComponentInvocation(
  declaration: RenderDeclarationAst,
): declaration is ComponentInvocationAst {
  return declaration.kind === "component-invocation";
}

function isIdentifierReference(
  value: ArgumentValueAst,
): value is IdentifierReferenceAst {
  return value.kind === "identifier-reference";
}

function symbolSchema(symbol: SymbolDefinitionAst): ComponentPropsSchema {
  return {
    id: symbol.name,
    props: Object.fromEntries(
      symbol.params.map((param) => [
        param.name,
        {
          kind: BUILTIN_PARAM_KINDS[param.type.name] ?? "json",
          required: true,
        },
      ]),
    ),
  };
}

function resolveValue(
  value: ArgumentValueAst,
  tokens: ReadonlyMap<string, JsonValue>,
  diagnostics: Diagnostic[],
): JsonValue | undefined {
  if (value.kind === "literal") {
    return value.value;
  }
  if (value.kind === "token-reference") {
    const resolved = tokens.get(value.token);
    if (resolved !== undefined) {
      return resolved;
    }
    diagnostics.push(
      diagnostic(
        "LINK_UNKNOWN_REFERENCE",
        "error",
        `Unknown token reference "${value.token}".`,
        value.sourceMap,
        `Declare \`token ${value.token} = "..."\` at the document level.`,
      ),
    );
    return undefined;
  }
  if (value.kind === "theme-role-reference") {
    diagnostics.push(
      diagnostic(
        "SYMBOL_INVALID_THEME_REFERENCE",
        "error",
        `Theme role "${value.role}" cannot be used as a component argument.`,
        value.sourceMap,
        "Use theme role references only in render style properties.",
      ),
    );
    return undefined;
  }
  if (isIdentifierReference(value)) {
    diagnostics.push(
      diagnostic(
        "SYMBOL_UNKNOWN_PARAMETER",
        "error",
        `Unknown symbol parameter "${value.name}".`,
        value.sourceMap,
      ),
    );
    return undefined;
  }
  if (value.kind !== "object-literal") {
    return undefined;
  }

  const object: Record<string, JsonValue> = {};
  for (const property of value.properties) {
    const resolved = resolveValue(property.value, tokens, diagnostics);
    if (resolved !== undefined) {
      object[property.name] = resolved;
    }
  }
  return object;
}

function substituteValue(
  value: ArgumentValueAst,
  bindings: Bindings,
  diagnostics: Diagnostic[],
): ArgumentValueAst {
  if (isIdentifierReference(value)) {
    const bound = bindings.get(value.name);
    if (bound !== undefined) {
      return bound;
    }
    diagnostics.push(
      diagnostic(
        "SYMBOL_UNKNOWN_PARAMETER",
        "error",
        `Unknown symbol parameter "${value.name}".`,
        value.sourceMap,
        `Declare "${value.name}" as a symbol parameter or replace the reference with a value.`,
      ),
    );
    return value;
  }
  if (value.kind !== "object-literal") {
    return value;
  }

  let changed = false;
  const properties = value.properties.map((property) => {
    const substituted = substituteValue(property.value, bindings, diagnostics);
    if (substituted === property.value) {
      return property;
    }
    changed = true;
    return {
      ...property,
      value: substituted as ValueAst | IdentifierReferenceAst,
    };
  });
  return changed ? { ...value, properties } : value;
}

function substituteProps(
  invocation: ComponentInvocationAst,
  bindings: Bindings,
  diagnostics: Diagnostic[],
  forceClone: boolean,
): ComponentInvocationAst {
  let changed = forceClone;
  const props = invocation.props.map((prop) => {
    const value = substituteValue(prop.value, bindings, diagnostics);
    if (value === prop.value) {
      return forceClone ? { ...prop } : prop;
    }
    changed = true;
    return { ...prop, value };
  });
  return changed ? { ...invocation, props } : invocation;
}

function validateBindings(
  invocation: ComponentInvocationAst,
  symbol: SymbolDefinitionAst,
  tokens: ReadonlyMap<string, JsonValue>,
  diagnostics: Diagnostic[],
): Bindings | undefined {
  const resolved: Record<string, JsonValue> = {};
  const bindings = new Map<string, ArgumentValueAst>();

  for (const prop of invocation.props) {
    const runtimeValue = resolveValue(prop.value, tokens, diagnostics);
    if (runtimeValue !== undefined) {
      resolved[prop.name] = runtimeValue;
      bindings.set(prop.name, prop.value);
    }
  }

  const validation = validateProps(resolved, symbolSchema(symbol), invocation.sourceMap);
  diagnostics.push(...validation.diagnostics);
  return hasErrors(diagnostics) ? undefined : bindings;
}

function unsupportedBodyDiagnostic(
  symbol: SymbolDefinitionAst,
  entry: SymbolDefinitionAst["body"][number],
): Diagnostic | undefined {
  if (entry.kind === "component-invocation") {
    return undefined;
  }
  return diagnostic(
    "SYMBOL_EXPANSION_UNSUPPORTED",
    "error",
    `Symbol "${symbol.name}" contains unsupported "${entry.kind}" body syntax.`,
    entry.sourceMap,
    "Use a flat body of component invocations; slots and loops are not supported.",
  );
}

function diagnoseSlots(
  invocation: ComponentInvocationAst,
  diagnostics: Diagnostic[],
): boolean {
  if (invocation.slots === undefined || invocation.slots.length === 0) {
    return false;
  }
  for (const slot of invocation.slots) {
    diagnostics.push(
      diagnostic(
        "SYMBOL_EXPANSION_UNSUPPORTED",
        "error",
        `Component invocation "${invocation.name}" contains unsupported slot "${slot.name}".`,
        slot.sourceMap,
        "Remove the slot; symbol expansion currently supports flat component calls only.",
      ),
    );
  }
  return true;
}

function expandInvocation(
  invocation: ComponentInvocationAst,
  symbols: SymbolTable,
  tokens: ReadonlyMap<string, JsonValue>,
  bindings: Bindings,
  stack: readonly string[],
  diagnostics: Diagnostic[],
  forceClone = false,
): readonly ComponentInvocationAst[] {
  const substituted = substituteProps(invocation, bindings, diagnostics, forceClone);
  if (hasErrors(diagnostics)) {
    return [];
  }

  if (diagnoseSlots(substituted, diagnostics)) {
    return [];
  }

  const symbol =
    substituted.namespace === undefined
      ? symbols.get(substituted.name)
      : undefined;
  if (symbol === undefined) {
    return [substituted];
  }

  if (stack.includes(symbol.name)) {
    diagnostics.push(
      diagnostic(
        "SYMBOL_EXPANSION_CYCLE",
        "error",
        `Recursive symbol expansion detected: ${[...stack, symbol.name].join(" -> ")}.`,
        substituted.sourceMap,
        "Remove the direct or indirect recursive symbol invocation.",
      ),
    );
    return [];
  }

  const nextBindings = validateBindings(substituted, symbol, tokens, diagnostics);
  if (nextBindings === undefined) {
    return [];
  }

  const expanded: ComponentInvocationAst[] = [];
  const nextStack = [...stack, symbol.name];
  for (const entry of symbol.body) {
    const unsupported = unsupportedBodyDiagnostic(symbol, entry);
    if (unsupported !== undefined) {
      diagnostics.push(unsupported);
      continue;
    }
    expanded.push(
      ...expandInvocation(
        entry as ComponentInvocationAst,
        symbols,
        tokens,
        nextBindings,
        nextStack,
        diagnostics,
        true,
      ),
    );
  }
  return expanded;
}

function expandRenderDeclarations(
  declarations: readonly RenderDeclarationAst[],
  symbols: SymbolTable,
  tokens: ReadonlyMap<string, JsonValue>,
  diagnostics: Diagnostic[],
): readonly RenderDeclarationAst[] {
  let changed = false;
  const expanded: RenderDeclarationAst[] = [];

  for (const declaration of declarations) {
    if (!isComponentInvocation(declaration)) {
      expanded.push(declaration);
      continue;
    }

    const next = expandInvocation(
      declaration,
      symbols,
      tokens,
      new Map(),
      [],
      diagnostics,
    );
    expanded.push(...next);
    changed ||= next.length !== 1 || next[0] !== declaration;
  }

  return changed ? expanded : declarations;
}

function expandScenes(
  scenes: readonly SceneAst[],
  symbols: SymbolTable,
  tokens: ReadonlyMap<string, JsonValue>,
  diagnostics: Diagnostic[],
): readonly SceneAst[] | null {
  let changed = false;
  const nextScenes = scenes.map((scene) => {
    const renderDeclarations = expandRenderDeclarations(
      scene.renderDeclarations,
      symbols,
      tokens,
      diagnostics,
    );
    if (renderDeclarations === scene.renderDeclarations) {
      return scene;
    }
    changed = true;
    return { ...scene, renderDeclarations };
  });
  return changed ? nextScenes : null;
}

/** Expands symbol invocations within a document against its symbol table. */
export function expandSymbolInvocations(
  document: DocumentAst,
  symbols: SymbolTable,
): Result<DocumentAst> {
  const diagnostics: Diagnostic[] = [];
  const tokens = new Map<string, JsonValue>(
    document.tokens.map((token) => [token.id, token.value.value]),
  );
  const scenes = expandScenes(document.scenes, symbols, tokens, diagnostics);
  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }

  if (scenes === null) {
    return { ok: true, value: document, diagnostics };
  }

  return {
    ok: true,
    value: { ...document, scenes },
    diagnostics,
  };
}
