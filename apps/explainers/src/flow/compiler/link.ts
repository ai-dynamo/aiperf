/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Symbol resolution for parsed `.flow` documents.
//!
//! Linking builds per-scene node symbol tables and a document-level token
//! table, then validates that every identifier reference (connector
//! endpoints, camera/timeline/interaction/responsive targets, reading-order
//! entries, and token references) resolves to a declared symbol. It also
//! flattens authored and imported theme declarations into a deterministic
//! validation set and rejects multiple `use theme` defaults. It does not
//! transform the AST; lowering consumes the resolved tables it produces.

import type {
  ComponentInvocationAst,
  ConnectorAst,
  DocumentAst,
  ImportDeclarationAst,
  LiteralAst,
  RectAst,
  RenderDeclarationAst,
  SceneAst,
  SymbolBodyStatementAst,
  ThemeDeclarationAst,
  UseThemeAst,
  ValueAst,
} from "../language/index.js";
import {
  diagnostic,
  hasErrors,
  type ComponentCatalog,
  type Diagnostic,
  type Result,
  type SourceRange,
} from "../schema/index.js";
import { findUnresolvedAfterRefs } from "./timeline-timing.js";

/** Node ids declared by a scene's render declarations, keyed for lookup. */
export type SceneSymbolTable = Readonly<{
  nodes: ReadonlyMap<string, RectAst | ConnectorAst>;
}>;

/** Compatibility name for the parsed namespace-import declaration. */
export type ModuleImportAst = ImportDeclarationAst;

/** A host-provided module made available to namespace-name resolution. */
export type ResolvedModule = Readonly<{
  canonicalUri: string;
  exports: ReadonlySet<string>;
  /** Theme declarations exported by the module, flattened into validation. */
  themes?: readonly ThemeDeclarationAst[];
  /** A `use theme` default declared by the module, if any. */
  useTheme?: UseThemeAst;
}>;

/** Input supplied to an injected module resolver. */
export type ModuleResolutionRequest = Readonly<{
  path: string;
  alias: string;
  importer: DocumentAst;
}>;

/**
 * Resolves one exact import path without ambient filesystem or network access.
 *
 * Returning `undefined` means that the configured build policy cannot resolve
 * the module.
 */
export type ModuleResolver = (
  request: ModuleResolutionRequest,
) => ResolvedModule | undefined;

/** Optional host services used while linking a parsed document. */
export type LinkOptions = Readonly<{
  resolveModule?: ModuleResolver;
  /**
   * When provided, unqualified invocations matching a catalog `id` or
   * `symbolExport` are treated as known (not `LINK_UNKNOWN_NAME`), so later
   * `validate(..., components)` can run prop / COMPONENT_UNKNOWN checks.
   */
  components?: ComponentCatalog;
}>;

/** Identity of an exported member reached through a namespace alias. */
export type ResolvedQualifiedName = Readonly<{
  canonicalUri: string;
  exportName: string;
}>;

/** A parsed document paired with all locally and externally resolved symbols. */
export type LinkedDocument = Readonly<{
  document: DocumentAst;
  tokens: ReadonlyMap<string, LiteralAst["value"]>;
  scenes: ReadonlyMap<string, SceneSymbolTable>;
  imports: ReadonlyMap<string, ResolvedModule>;
  qualifiedNames: ReadonlyMap<ComponentInvocationAst, ResolvedQualifiedName>;
  /** Authored plus imported theme declarations, in primary-then-import order. */
  themes: readonly ThemeDeclarationAst[];
  /** The single selected default theme across the linked document, if any. */
  useTheme?: UseThemeAst;
}>;

type Identified = Readonly<{ id: string; sourceMap: SourceRange }>;

function identifiedRenderNodes(
  declarations: readonly RenderDeclarationAst[],
): readonly (RectAst | ConnectorAst)[] {
  return declarations.filter(
    (node): node is RectAst | ConnectorAst =>
      node.kind === "rect" || node.kind === "connector",
  );
}

function duplicateIdDiagnostics<T extends Identified>(
  items: readonly T[],
  message: (id: string) => string,
): readonly Diagnostic[] {
  const seen = new Set<string>();
  const diagnostics: Diagnostic[] = [];
  for (const item of items) {
    if (seen.has(item.id)) {
      diagnostics.push(
        diagnostic("LINK_DUPLICATE_ID", "error", message(item.id), item.sourceMap),
      );
      continue;
    }
    seen.add(item.id);
  }
  return diagnostics;
}

function checkReference(
  targetId: string,
  nodes: ReadonlyMap<string, RectAst | ConnectorAst>,
  range: SourceRange,
  what: string,
): Diagnostic | undefined {
  if (nodes.has(targetId)) {
    return undefined;
  }
  return diagnostic(
    "LINK_UNKNOWN_REFERENCE",
    "error",
    `Unknown ${what} "${targetId}".`,
    range,
    `Declare a node named "${targetId}" in this scene or fix the reference.`,
  );
}

/**
 * Reference check that also accepts component-instance ids. Timeline cues,
 * cameras, interactions, responsive overrides, and reading-order entries may
 * target a native component instance (before SDK expansion) as well as a raw
 * render node.
 */
function checkReferenceId(
  targetId: string,
  ids: ReadonlySet<string>,
  range: SourceRange,
  what: string,
): Diagnostic | undefined {
  if (ids.has(targetId)) {
    return undefined;
  }
  return diagnostic(
    "LINK_UNKNOWN_REFERENCE",
    "error",
    `Unknown ${what} "${targetId}".`,
    range,
    `Declare a node or component instance named "${targetId}" in this scene or fix the reference.`,
  );
}

/** Reads the literal `id` prop of a component invocation, if present. */
function componentInstanceId(
  invocation: ComponentInvocationAst,
): string | undefined {
  const idProp = invocation.props.find((prop) => prop.name === "id");
  if (idProp === undefined) {
    return undefined;
  }
  const { value } = idProp;
  return value.kind === "literal" && typeof value.value === "string"
    ? value.value
    : undefined;
}

/**
 * Collects component-instance ids that a scene exposes as reference targets:
 * top-level component invocations and top-level invocations inside `freeform`
 * blocks. `for`-loop bodies are excluded because their per-iteration instance
 * ids are produced during SDK expansion, not authored directly.
 */
function collectComponentInstanceIds(scene: SceneAst): readonly string[] {
  const ids: string[] = [];
  const fromDeclarations = (
    declarations: readonly RenderDeclarationAst[],
  ): void => {
    for (const declaration of declarations) {
      if (declaration.kind === "component-invocation") {
        const id = componentInstanceId(declaration);
        if (id !== undefined) {
          ids.push(id);
        }
      }
    }
  };
  fromDeclarations(scene.renderDeclarations);
  for (const freeform of scene.freeforms ?? []) {
    fromDeclarations(freeform.body);
  }
  return ids;
}

function checkValueReference(
  value: ValueAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): Diagnostic | undefined {
  if (value.kind !== "token-reference" || tokens.has(value.token)) {
    return undefined;
  }
  return diagnostic(
    "LINK_UNKNOWN_REFERENCE",
    "error",
    `Unknown token reference "${value.token}".`,
    value.sourceMap,
    `Declare \`token ${value.token} = "..."\` at the document level.`,
  );
}

function invalidImportDiagnostic(
  declaration: ImportDeclarationAst,
): Diagnostic | undefined {
  const { path } = declaration;
  if (path.startsWith("https://")) {
    return diagnostic(
      "MODULE_INTEGRITY_REQUIRED",
      "error",
      `Remote import "${path}" requires an integrity digest.`,
      declaration.sourceMap,
      "Add verified integrity metadata before resolving this remote module.",
    );
  }
  if (
    path.startsWith("./") ||
    path.startsWith("../") ||
    path.startsWith("@")
  ) {
    return undefined;
  }
  return diagnostic(
    "MODULE_INVALID_SPECIFIER",
    "error",
    `Invalid module import specifier "${path}".`,
    declaration.sourceMap,
    "Use an exact relative .flow path or a configured package import.",
  );
}

function moduleNotFoundDiagnostic(
  declaration: ImportDeclarationAst,
): Diagnostic {
  return diagnostic(
    "MODULE_NOT_FOUND",
    "error",
    `Unable to resolve module "${declaration.path}".`,
    declaration.sourceMap,
    "Configure a module resolver that provides this exact import.",
  );
}

function resolveImports(
  document: DocumentAst,
  options: LinkOptions,
): Readonly<{
  imports: ReadonlyMap<string, ResolvedModule>;
  declaredAliases: ReadonlySet<string>;
  diagnostics: readonly Diagnostic[];
}> {
  const imports = new Map<string, ResolvedModule>();
  const declaredAliases = new Set<string>();
  const localBindings = new Set<string>([
    ...document.tokens.map(({ id }) => id),
    ...document.symbols.map(({ name }) => name),
    ...document.scenes.map(({ id }) => id),
  ]);
  const diagnostics: Diagnostic[] = [];

  for (const declaration of document.imports ?? []) {
    if (
      declaredAliases.has(declaration.alias) ||
      localBindings.has(declaration.alias)
    ) {
      diagnostics.push(
        diagnostic(
          "LINK_DUPLICATE_BINDING",
          "error",
          `Duplicate import alias "${declaration.alias}".`,
          declaration.sourceMap,
          "Choose a unique alias for this module import.",
        ),
      );
      declaredAliases.add(declaration.alias);
      continue;
    }
    declaredAliases.add(declaration.alias);

    const invalid = invalidImportDiagnostic(declaration);
    if (invalid !== undefined) {
      diagnostics.push(invalid);
      continue;
    }

    let resolved: ResolvedModule | undefined;
    try {
      resolved = options.resolveModule?.({
        path: declaration.path,
        alias: declaration.alias,
        importer: document,
      });
    } catch {
      resolved = undefined;
    }
    if (
      resolved === undefined ||
      resolved.canonicalUri.length === 0
    ) {
      diagnostics.push(moduleNotFoundDiagnostic(declaration));
      continue;
    }
    imports.set(declaration.alias, resolved);
  }

  return { imports, declaredAliases, diagnostics };
}

function collectComponentInvocations(
  statements: readonly SymbolBodyStatementAst[],
  output: ComponentInvocationAst[],
): void {
  for (const statement of statements) {
    if (statement.kind === "component-invocation") {
      output.push(statement);
      for (const slot of statement.slots ?? []) {
        collectComponentInvocations(slot.body, output);
      }
      continue;
    }
    collectComponentInvocations(statement.body, output);
  }
}

function localComponentNames(
  document: DocumentAst,
  components?: ComponentCatalog,
): ReadonlySet<string> {
  const names = new Set(document.symbols.map(({ name }) => name));
  if (components !== undefined) {
    for (const descriptor of components.components) {
      names.add(descriptor.id);
      names.add(descriptor.symbolExport);
    }
  }
  return names;
}

function qualifiedNameDiagnostics(
  document: DocumentAst,
  imports: ReadonlyMap<string, ResolvedModule>,
  declaredAliases: ReadonlySet<string>,
  components?: ComponentCatalog,
): Readonly<{
  qualifiedNames: ReadonlyMap<ComponentInvocationAst, ResolvedQualifiedName>;
  diagnostics: readonly Diagnostic[];
}> {
  const invocations: ComponentInvocationAst[] = [];
  for (const symbol of document.symbols) {
    collectComponentInvocations(symbol.body, invocations);
  }
  for (const scene of document.scenes) {
    for (const declaration of scene.renderDeclarations) {
      if (declaration.kind === "component-invocation") {
        invocations.push(declaration);
        for (const slot of declaration.slots ?? []) {
          collectComponentInvocations(slot.body, invocations);
        }
      }
    }
    for (const loop of scene.loops ?? []) {
      collectComponentInvocations(loop.body, invocations);
    }
    for (const freeform of scene.freeforms ?? []) {
      for (const declaration of freeform.body) {
        if (declaration.kind === "component-invocation") {
          invocations.push(declaration);
          for (const slot of declaration.slots ?? []) {
            collectComponentInvocations(slot.body, invocations);
          }
        }
      }
    }
  }
  invocations.sort(
    (left, right) =>
      left.sourceMap.start.offset - right.sourceMap.start.offset ||
      left.sourceMap.end.offset - right.sourceMap.end.offset,
  );

  const locals = localComponentNames(document, components);
  const qualifiedNames = new Map<
    ComponentInvocationAst,
    ResolvedQualifiedName
  >();
  const diagnostics: Diagnostic[] = [];
  for (const invocation of invocations) {
    if (invocation.namespace === undefined) {
      if (!locals.has(invocation.name)) {
        diagnostics.push(
          diagnostic(
            "LINK_UNKNOWN_NAME",
            "error",
            `Unknown component or symbol "${invocation.name}".`,
            invocation.sourceMap,
            `Declare \`symbol ${invocation.name}(...)\` in this document, register it in the component catalog, or fix the name.`,
          ),
        );
      }
      continue;
    }
    if (!declaredAliases.has(invocation.namespace)) {
      diagnostics.push(
        diagnostic(
          "LINK_UNKNOWN_NAME",
          "error",
          `Unknown import namespace "${invocation.namespace}".`,
          invocation.sourceMap,
          `Import a module as "${invocation.namespace}" or fix the qualified name.`,
        ),
      );
      continue;
    }

    const imported = imports.get(invocation.namespace);
    if (imported === undefined) {
      continue;
    }
    if (!imported.exports.has(invocation.name)) {
      diagnostics.push(
        diagnostic(
          "LINK_UNKNOWN_NAMESPACE_MEMBER",
          "error",
          `Module namespace "${invocation.namespace}" does not export "${invocation.name}".`,
          invocation.sourceMap,
          "Use an exported member or update the imported module.",
        ),
      );
      continue;
    }
    qualifiedNames.set(invocation, {
      canonicalUri: imported.canonicalUri,
      exportName: invocation.name,
    });
  }

  return { qualifiedNames, diagnostics };
}

/** Resolves symbol tables for a parsed document and diagnoses broken references. */
export function link(
  document: DocumentAst,
  options: LinkOptions = {},
): Result<LinkedDocument> {
  const moduleResolution = resolveImports(document, options);
  const qualifiedResolution = qualifiedNameDiagnostics(
    document,
    moduleResolution.imports,
    moduleResolution.declaredAliases,
    options.components,
  );
  const diagnostics: Diagnostic[] = [
    ...moduleResolution.diagnostics,
    ...qualifiedResolution.diagnostics,
    ...duplicateIdDiagnostics(
      document.scenes,
      (id) => `Duplicate scene id "${id}".`,
    ),
    ...duplicateIdDiagnostics(
      document.tokens,
      (id) => `Duplicate token id "${id}".`,
    ),
  ];

  const tokens = new Map<string, LiteralAst["value"]>(
    document.tokens.map((token) => [token.id, token.value.value]),
  );

  // Flatten in primary-then-import declaration order so validation and later
  // lowering see a stable theme list independent of Map iteration quirks.
  // Deduplicate by canonicalUri so importing one module under two aliases
  // does not double-count themes / use-theme defaults.
  const importedModules: ResolvedModule[] = [];
  const seenCanonicalUris = new Set<string>();
  for (const declaration of document.imports ?? []) {
    const resolved = moduleResolution.imports.get(declaration.alias);
    if (resolved === undefined) {
      continue;
    }
    if (seenCanonicalUris.has(resolved.canonicalUri)) {
      continue;
    }
    seenCanonicalUris.add(resolved.canonicalUri);
    importedModules.push(resolved);
  }
  const themes: readonly ThemeDeclarationAst[] = [
    ...(document.themes ?? []),
    ...importedModules.flatMap((module) => module.themes ?? []),
  ];
  const useThemes: readonly UseThemeAst[] = [
    ...(document.useTheme === undefined ? [] : [document.useTheme]),
    ...importedModules.flatMap((module) =>
      module.useTheme === undefined ? [] : [module.useTheme],
    ),
  ];
  for (const extra of useThemes.slice(1)) {
    diagnostics.push(
      diagnostic(
        "THEME_DUPLICATE_DEFAULT",
        "error",
        `More than one "use theme" default is declared for the linked document.`,
        extra.sourceMap,
        "Declare a single default theme across the document and its imports.",
      ),
    );
  }

  const scenes = new Map<string, SceneSymbolTable>();
  for (const scene of document.scenes) {
    diagnostics.push(
      ...duplicateIdDiagnostics(
        identifiedRenderNodes(scene.renderDeclarations),
        (id) => `Duplicate node id "${id}" in scene "${scene.id}".`,
      ),
      ...duplicateIdDiagnostics(
        scene.cameras,
        (id) => `Duplicate camera id "${id}" in scene "${scene.id}".`,
      ),
      ...duplicateIdDiagnostics(
        scene.timelines,
        (id) => `Duplicate timeline id "${id}" in scene "${scene.id}".`,
      ),
      ...duplicateIdDiagnostics(
        scene.interactions,
        (id) => `Duplicate interaction id "${id}" in scene "${scene.id}".`,
      ),
      ...duplicateIdDiagnostics(
        scene.responsiveVariants,
        (id) => `Duplicate responsive id "${id}" in scene "${scene.id}".`,
      ),
    );

    const nodes = new Map<string, RectAst | ConnectorAst>();
    for (const node of identifiedRenderNodes(scene.renderDeclarations)) {
      if (!nodes.has(node.id)) {
        nodes.set(node.id, node);
      }
    }
    scenes.set(scene.id, { nodes });
  }

  for (const scene of document.scenes) {
    const nodes = scenes.get(scene.id)?.nodes ?? new Map<string, RectAst | ConnectorAst>();
    const referenceableIds = new Set<string>([
      ...nodes.keys(),
      ...collectComponentInstanceIds(scene),
    ]);

    for (const node of scene.renderDeclarations) {
      if (node.kind === "component-invocation") {
        continue;
      }
      if (node.kind === "connector") {
        diagnostics.push(
          ...[
            checkReference(node.from, nodes, node.sourceMap, "connector `from`"),
            checkReference(node.to, nodes, node.sourceMap, "connector `to`"),
          ].filter((entry): entry is Diagnostic => entry !== undefined),
        );
      }
      if (node.kind === "scene-primitive") {
        continue;
      }
      const value = node.kind === "rect" ? node.fill : node.stroke;
      const valueDiagnostic = checkValueReference(value, tokens);
      if (valueDiagnostic !== undefined) {
        diagnostics.push(valueDiagnostic);
      }
    }

    for (const camera of scene.cameras) {
      for (const keyframe of camera.keyframes) {
        for (const target of keyframe.targets.references) {
          const targetDiagnostic = checkReferenceId(
            target,
            referenceableIds,
            keyframe.sourceMap,
            "camera target",
          );
          if (targetDiagnostic !== undefined) {
            diagnostics.push(targetDiagnostic);
          }
        }
      }
    }

    for (const timeline of scene.timelines) {
      for (const cue of timeline.cues) {
        const targetDiagnostic = checkReferenceId(
          cue.target,
          referenceableIds,
          cue.sourceMap,
          "timeline target",
        );
        if (targetDiagnostic !== undefined) {
          diagnostics.push(targetDiagnostic);
        }
      }
      for (const index of findUnresolvedAfterRefs(timeline.cues)) {
        const cue = timeline.cues[index]!;
        if (cue.timing.mode === "after") {
          diagnostics.push(
            diagnostic(
              "LINK_UNKNOWN_REFERENCE",
              "error",
              `Timeline "${timeline.id}" cue "after ${cue.timing.ref}" does not match any earlier cue's target in this timeline.`,
              cue.sourceMap,
            ),
          );
        }
      }
    }

    for (const interaction of scene.interactions) {
      diagnostics.push(
        ...[
          checkReferenceId(
            interaction.event.target,
            referenceableIds,
            interaction.sourceMap,
            "interaction event target",
          ),
          checkReferenceId(
            interaction.action.target,
            referenceableIds,
            interaction.sourceMap,
            "interaction action target",
          ),
        ].filter((entry): entry is Diagnostic => entry !== undefined),
      );
    }

    for (const responsive of scene.responsiveVariants) {
      for (const override of responsive.overrides) {
        const targetDiagnostic = checkReferenceId(
          override.target,
          referenceableIds,
          override.sourceMap,
          "responsive override target",
        );
        if (targetDiagnostic !== undefined) {
          diagnostics.push(targetDiagnostic);
        }
      }
    }

    if (scene.readingOrder !== undefined) {
      for (const reference of scene.readingOrder.references) {
        const referenceDiagnostic = checkReferenceId(
          reference,
          referenceableIds,
          scene.readingOrder.sourceMap,
          "reading-order reference",
        );
        if (referenceDiagnostic !== undefined) {
          diagnostics.push(referenceDiagnostic);
        }
      }
    }
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return {
    ok: true,
    value: {
      document,
      tokens,
      scenes,
      imports: moduleResolution.imports,
      qualifiedNames: qualifiedResolution.qualifiedNames,
      themes,
      ...(useThemes[0] === undefined ? {} : { useTheme: useThemes[0] }),
    },
    diagnostics: [],
  };
}
