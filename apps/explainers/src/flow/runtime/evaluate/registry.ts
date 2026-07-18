// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral capability evaluator registration.

import type { ComponentNodeIr, LayoutPlanIr } from "../../schema/index.js";

import type { DrawCommand, HitRegion } from "../display-list.js";
import type {
  SemanticEntityProjection,
  SemanticRelationProjection,
} from "./types.js";

/** Deterministic inputs available to a capability evaluator. */
export type CapabilityEvaluationContext = Readonly<{
  atMs: number;
}>;

/** Display-list fragment emitted by one capability evaluator. */
export type CapabilityDisplayContribution = Readonly<{
  commands: readonly DrawCommand[];
  hitRegions: readonly HitRegion[];
}>;

/** Semantic fragment emitted by one capability evaluator. */
export type CapabilitySemanticContribution = Readonly<{
  entities: readonly SemanticEntityProjection[];
  relations: readonly SemanticRelationProjection[];
  readingOrder: readonly string[];
}>;

/** Immutable backend-neutral products emitted for one component node. */
export type CapabilityContribution = Readonly<{
  layout?: LayoutPlanIr | undefined;
  display: CapabilityDisplayContribution;
  semantic: CapabilitySemanticContribution;
}>;

/** Evaluates component IR without depending on a rendering backend. */
export type CapabilityEvaluator<
  TContribution extends CapabilityContribution = CapabilityContribution,
> = Readonly<{
  capabilityId: string;
  evaluate(
    node: ComponentNodeIr,
    context: CapabilityEvaluationContext,
  ): TContribution;
}>;

/** Raised when an evaluator id appears more than once. */
export class DuplicateCapabilityEvaluatorError extends Error {
  constructor(capabilityId: string) {
    super(`Capability evaluator "${capabilityId}" is already registered.`);
    this.name = "DuplicateCapabilityEvaluatorError";
  }
}

/** Raised when evaluation requests an unregistered capability id. */
export class UnknownCapabilityEvaluatorError extends Error {
  constructor(capabilityId: string) {
    super(`No capability evaluator is registered for "${capabilityId}".`);
    this.name = "UnknownCapabilityEvaluatorError";
  }
}

/** Immutable evaluator lookup used by scene evaluation. */
export class FrozenCapabilityEvaluatorRegistry {
  readonly #evaluators: ReadonlyMap<string, CapabilityEvaluator>;
  readonly #capabilityIds: readonly string[];

  constructor(evaluators: ReadonlyMap<string, CapabilityEvaluator>) {
    this.#evaluators = new Map(evaluators);
    this.#capabilityIds = Object.freeze(
      [...this.#evaluators.keys()].sort((left, right) =>
        left.localeCompare(right),
      ),
    );
    Object.freeze(this);
  }

  /** Returns the evaluator or `undefined` when the capability is unknown. */
  get(capabilityId: string): CapabilityEvaluator | undefined {
    return this.#evaluators.get(capabilityId);
  }

  /** Returns the evaluator and fails closed when the capability is unknown. */
  require(capabilityId: string): CapabilityEvaluator {
    const evaluator = this.get(capabilityId);
    if (evaluator === undefined) {
      throw new UnknownCapabilityEvaluatorError(capabilityId);
    }
    return evaluator;
  }

  /** Returns registered ids in deterministic lexical order. */
  capabilityIds(): readonly string[] {
    return this.#capabilityIds;
  }
}

/** Transactional builder for an immutable evaluator lookup. */
export class CapabilityEvaluatorRegistry {
  readonly #evaluators = new Map<string, CapabilityEvaluator>();
  #frozen = false;

  constructor(evaluators: readonly CapabilityEvaluator[] = []) {
    this.registerAll(evaluators);
  }

  /** Registers one evaluator. */
  register(evaluator: CapabilityEvaluator): void {
    this.registerAll([evaluator]);
  }

  /** Registers a batch atomically after checking every id for duplicates. */
  registerAll(evaluators: readonly CapabilityEvaluator[]): void {
    if (this.#frozen) {
      throw new Error("Capability evaluator registry is frozen.");
    }

    const pendingIds = new Set<string>();
    for (const evaluator of evaluators) {
      const { capabilityId } = evaluator;
      if (
        this.#evaluators.has(capabilityId) ||
        pendingIds.has(capabilityId)
      ) {
        throw new DuplicateCapabilityEvaluatorError(capabilityId);
      }
      pendingIds.add(capabilityId);
    }

    for (const evaluator of evaluators) {
      this.#evaluators.set(evaluator.capabilityId, evaluator);
    }
  }

  /** Seals registration and returns an immutable lookup snapshot. */
  freeze(): FrozenCapabilityEvaluatorRegistry {
    this.#frozen = true;
    return new FrozenCapabilityEvaluatorRegistry(this.#evaluators);
  }
}
