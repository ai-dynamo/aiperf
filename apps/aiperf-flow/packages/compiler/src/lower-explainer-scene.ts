/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower an embedded slide `@scene` AST node to a DeckPackage `SceneRender`.
//!
//! Reuses the document scene lowerer via a single-scene `LinkedDocument`, then
//! fail-closes with `sceneIrSchema` so invalid scenes never become package
//! render payloads.

import type {
  ConnectorAst,
  DocumentAst,
  LiteralAst,
  RectAst,
  SceneAst,
} from "@aiperf/flow-language";
import {
  diagnostic,
  sceneIrSchema,
  type Result,
  type SceneRender,
  type SourceRange,
} from "@aiperf/flow-schema";

import type { LinkedDocument, SceneSymbolTable } from "./link.js";
import { lower } from "./lower.js";

export type LowerExplainerSceneOptions = Readonly<{
  tokens?: ReadonlyMap<string, LiteralAst["value"]>;
}>;

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

function isSceneAst(value: unknown): value is SceneAst {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "scene" &&
    "id" in value &&
    typeof value.id === "string" &&
    "title" in value &&
    typeof value.title === "string" &&
    "renderDeclarations" in value &&
    Array.isArray(value.renderDeclarations)
  );
}

function sceneSymbolTable(scene: SceneAst): SceneSymbolTable {
  const nodes = new Map<string, RectAst | ConnectorAst>();
  for (const node of scene.renderDeclarations) {
    if (
      (node.kind === "rect" || node.kind === "connector") &&
      !nodes.has(node.id)
    ) {
      nodes.set(node.id, node);
    }
  }
  return { nodes };
}

function wrapSceneDocument(
  scene: SceneAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): LinkedDocument {
  const document: DocumentAst = {
    kind: "document",
    id: `explainer-scene-${scene.id}`,
    title: scene.title,
    language: { kind: "language", version: 1, sourceMap: scene.sourceMap },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [scene],
    sourceMap: scene.sourceMap,
  };

  return {
    document,
    tokens,
    scenes: new Map([[scene.id, sceneSymbolTable(scene)]]),
    imports: new Map(),
    qualifiedNames: new Map(),
    themes: [],
  };
}

function invalidScene(
  message: string,
  range: SourceRange = unknownRange,
): Result<SceneRender> {
  return {
    ok: false,
    diagnostics: [
      diagnostic("EXPLAINER_SCENE_INVALID", "error", message, range),
    ],
  };
}

/**
 * Lowers an embedded slide `@scene` AST node to `{ kind: "scene", scene }`.
 *
 * Returns diagnostics when the input is not a scene AST or the lowered IR
 * fails strict `SceneIr` validation.
 */
export function lowerExplainerScene(
  scene: unknown,
  options: LowerExplainerSceneOptions = {},
): Result<SceneRender> {
  if (!isSceneAst(scene)) {
    return invalidScene(
      "Expected an embedded @scene AST node with kind \"scene\".",
    );
  }

  let lowered;
  try {
    lowered = lower(wrapSceneDocument(scene, options.tokens ?? new Map()));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Scene lowering failed.";
    return invalidScene(message, scene.sourceMap);
  }

  const sceneIr = lowered.scenes[0];
  if (sceneIr === undefined) {
    return invalidScene(
      `Scene "${scene.id}" produced no SceneIr after lowering.`,
      scene.sourceMap,
    );
  }

  const parsed = sceneIrSchema.safeParse(sceneIr);
  if (!parsed.success) {
    return {
      ok: false,
      diagnostics: parsed.error.issues.map((issue) => {
        const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
        return diagnostic(
          "EXPLAINER_SCENE_INVALID",
          "error",
          `${path}: ${issue.message}`,
          scene.sourceMap,
        );
      }),
    };
  }

  return {
    ok: true,
    value: { kind: "scene", scene: parsed.data },
    diagnostics: [],
  };
}
