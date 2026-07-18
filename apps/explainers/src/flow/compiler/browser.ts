/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Browser-safe public surface for Flow compilation and serialization.

export * from "./compile-source.js";
export {
  compileExplainerSource,
  type CompileExplainerRequest,
  type ExplainerCompileOptions,
} from "./compile-explainer.js";
export * from "./components.js";
export { formatDiagnostic } from "../diagnostics.js";
export * from "./expand-symbols.js";
export {
  link,
  type LinkedDocument,
  type LinkOptions,
  type ModuleImportAst,
  type ModuleResolver,
  type ModuleResolutionRequest as LinkModuleResolutionRequest,
  type ResolvedModule as LinkResolvedModule,
  type ResolvedQualifiedName,
  type SceneSymbolTable,
} from "./link.js";
export * from "./lower.js";
export {
  lowerExplainerToDeckPackage,
  type ExplainerDeckMetadata,
  type ExplainerLowerInput,
  type ExplainerLowerOptions,
} from "./lower-explainer.js";
export * from "./lower-explainer-scene.js";
export {
  lowerExplainerSlides,
  slideIdFromTitle,
  type SlideTextAst,
} from "./lower-explainer-slides.js";
export {
  canonicalizeModuleUri,
  resolveModuleGraph,
  type InjectedModuleSource,
  type ModuleImport,
  type ModuleManifestEntry,
  type ModuleResolutionRequest,
  type ModuleSourceKind,
  type PackageResolution,
  type ResolvedModule,
  type ResolvedModuleDependency,
  type ResolvedModuleGraph,
} from "./module-resolution.js";
export * from "./pack.js";
export { packDeckPackageToJson } from "./serialization.js";
export * from "./symbols.js";
export * from "./themes.js";
export * from "./validate.js";
export * from "./validate-explainer-set.js";
export * from "./validate-explainer-timelines.js";
