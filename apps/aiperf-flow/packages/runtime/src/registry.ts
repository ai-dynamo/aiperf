// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  CapabilityDescriptor,
  CapabilityRegistryManifest,
  RenderNodeIr,
  SceneIr,
} from "@aiperf/flow-schema";
import type { ReactNode } from "react";

import type { SceneAction, SceneState } from "./store.js";

export type RenderContext = Readonly<{
  state: SceneState;
  timeline: SceneIr["timeline"];
  dispatch(action: SceneAction): void;
  activateNode(nodeId: string): void;
  renderNode(node: RenderNodeIr): ReactNode;
  nodeById(id: string): RenderNodeIr | undefined;
}>;

export type RuntimeCapability<TNode extends RenderNodeIr = RenderNodeIr> =
  Readonly<{
    descriptor: CapabilityDescriptor;
    render(node: TNode, context: RenderContext): ReactNode;
  }>;

export class DuplicateCapabilityError extends Error {
  constructor(id: string) {
    super(`Runtime capability "${id}" is already registered.`);
    this.name = "DuplicateCapabilityError";
  }
}

export class MissingCapabilityError extends Error {
  constructor(id: string) {
    super(`Runtime capability "${id}" is not registered.`);
    this.name = "MissingCapabilityError";
  }
}

export class CapabilityRegistry {
  readonly #capabilities = new Map<string, RuntimeCapability>();

  register(capability: RuntimeCapability): void {
    const { id } = capability.descriptor;
    if (this.#capabilities.has(id)) {
      throw new DuplicateCapabilityError(id);
    }
    this.#capabilities.set(id, capability);
  }

  require(id: string): RuntimeCapability {
    const capability = this.#capabilities.get(id);
    if (capability === undefined) {
      throw new MissingCapabilityError(id);
    }
    return capability;
  }

  manifest(): CapabilityRegistryManifest {
    return {
      capabilities: [...this.#capabilities.values()]
        .map(({ descriptor }) => descriptor)
        .sort((left, right) => left.id.localeCompare(right.id)),
    } as CapabilityRegistryManifest;
  }
}
