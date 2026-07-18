// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export * from "./ast.js";
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
export { parseDocument, parseNativeEmbeddedScene } from "./parser.js";
export {
  parseExplainerBlock,
  parseSceneBlock,
  type ExplainerAstCompat,
  type SlideAstCompat,
  type TokenStream,
} from "./grammar/explainer.js";
