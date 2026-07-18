/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Strict prop validation for component-kind render nodes.
//!
//! `ComponentNodeIr.props` is an untyped JSON record at the IR boundary. This
//! module checks authored props against a component's declared prop schema.

import {
  diagnostic,
  hasErrors,
  type ComponentDescriptor,
  type ComponentPropDescriptor,
  type Diagnostic,
  type JsonValue,
  type Result,
  type SourceRange,
} from "../schema/index.js";

export const PROP_VALUE_KINDS = ["string", "number", "boolean", "json"] as const;

/** The runtime type a prop value must satisfy; `"json"` accepts any JSON value. */
export type PropValueKind = (typeof PROP_VALUE_KINDS)[number];

/** A single prop's type and requiredness within a component's schema. */
export type PropDescriptor = Readonly<{
  kind: PropValueKind;
  required: boolean;
}>;

/** A descriptor-like schema: a component id plus its declared prop contract. */
export type ComponentPropsSchema = Readonly<{
  id: string;
  props: Readonly<Record<string, PropDescriptor>>;
}>;

/** Canonical schema projection accepted from the shared Flow schema. */
export type DescriptorBackedComponentPropsSchema = Pick<
  ComponentDescriptor,
  "id" | "props"
>;

type ValidationPropDescriptor = PropDescriptor | ComponentPropDescriptor;

function descriptorKind(descriptor: ValidationPropDescriptor): string {
  return "type" in descriptor ? descriptor.type : descriptor.kind;
}

function matchesKind(value: JsonValue, kind: string): boolean {
  switch (kind) {
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number";
    case "boolean":
      return typeof value === "boolean";
    case "json":
      return true;
    default:
      return false;
  }
}

/** Validates authored props against a component's strict prop schema. */
export function validateProps(
  props: Readonly<Record<string, JsonValue>>,
  schema: ComponentPropsSchema | DescriptorBackedComponentPropsSchema,
  range: SourceRange,
): Result<Readonly<Record<string, JsonValue>>> {
  const diagnostics: Diagnostic[] = [];

  for (const key of Object.keys(props)) {
    if (key in schema.props) {
      continue;
    }
    diagnostics.push(
      diagnostic(
        "STRICT_UNKNOWN_PROP",
        "error",
        `Component "${schema.id}" does not declare a prop named "${key}".`,
        range,
        `Remove "${key}" or add it to the "${schema.id}" prop schema.`,
      ),
    );
  }

  for (const [name, descriptor] of Object.entries(schema.props)) {
    if (!descriptor.required || name in props) {
      continue;
    }
    diagnostics.push(
      diagnostic(
        "PROP_MISSING_REQUIRED",
        "error",
        `Component "${schema.id}" is missing required prop "${name}".`,
        range,
        `Add a value for "${name}".`,
      ),
    );
  }

  for (const [name, descriptor] of Object.entries(schema.props)) {
    const value = props[name];
    const kind = descriptorKind(descriptor);
    if (value === undefined || matchesKind(value, kind)) {
      continue;
    }
    diagnostics.push(
      diagnostic(
        "PROP_TYPE_MISMATCH",
        "error",
        `Prop "${name}" on component "${schema.id}" expects type "${kind}".`,
        range,
      ),
    );
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return { ok: true, value: props, diagnostics };
}
