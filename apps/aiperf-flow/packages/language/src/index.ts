// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public language surface for the Flow compiler and explainer build script:
//! `parseDocument`, embedded-scene helpers, and explainer grammar entry points.

export const FLOW_LANGUAGE_VERSION = 1 as const;

export * from "./ast.js";
/** Embedded `@scene` dialect helpers used by explainer lowering. */
export {
  captureEmbeddedScene,
  captureSceneBody,
  detectEmbeddedSceneForm,
  packageSceneToSceneAst,
  parseEmbeddedSceneSource,
  parsePackageSceneBody,
  type EmbeddedSceneForm,
  type EmbeddedSceneSource,
  type PackageSceneAst,
  type PackageSceneIrAst,
  type PeekableTokenStream,
} from "./embedded-scene.js";
export { formatDocument } from "./formatter.js";
/** Top-level `.flow` parse entry, including `explainer` documents. */
export { parseDocument, parseNativeEmbeddedScene } from "./parser.js";
export {
  parseExplainerBlock,
  parseSceneBlock,
  type ExplainerAstCompat,
  type SlideAstCompat,
  type TokenStream,
} from "./grammar/explainer.js";
