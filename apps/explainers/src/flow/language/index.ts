// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const FLOW_LANGUAGE_VERSION = 1 as const;

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
export { parseDocument, parseNativeEmbeddedScene } from "./parser.js";
