/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ComponentDescriptor } from "../schema/component-descriptor.js";
import type { Result } from "../schema/diagnostic.js";
import type {
  ConnectorEndpointIr,
  RenderNodeIr,
} from "../schema/ir.js";
import type { JsonValue } from "../schema/json-value.js";
import type { SourceRange } from "../schema/source.js";

/** Public timeline actions exposed by SDK component instances. */
export const SDK_ACTION_NAMES = [
  "enter",
  "draw",
  "trace",
  "emphasis",
  "pulse",
  "stagger",
  "exit",
  "fade",
] as const;

export type SdkActionName = (typeof SDK_ACTION_NAMES)[number];

/** Deterministic IR fragment produced by an SDK factory expansion. */
export type SceneFragment = Readonly<{
  roots: readonly RenderNodeIr[];
  ports: Readonly<Record<string, ConnectorEndpointIr>>;
  actions: Readonly<Partial<Record<SdkActionName, readonly string[]>>>;
}>;

/** Per-instance context passed to SDK factories during compile-time expansion. */
export type SdkExpansionContext = Readonly<{
  instanceId: string;
  sourceMap: SourceRange;
  themeTokens: ReadonlyMap<string, JsonValue>;
}>;

/** Pure compile-time factory that expands authored props and slots into Scene IR. */
export type SdkComponentFactory = (
  props: Readonly<Record<string, JsonValue>>,
  slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
) => Result<SceneFragment>;

/** Registry entry pairing a component descriptor with its factory and actions. */
export type SdkComponentDefinition = Readonly<{
  descriptor: ComponentDescriptor;
  factory: SdkComponentFactory;
  actions: readonly SdkActionName[];
}>;
