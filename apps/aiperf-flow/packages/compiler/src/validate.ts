/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Semantic validation for linked `.flow` documents.
//!
//! Validation runs after linking and before lowering. It checks that every
//! `require`d capability is registered, that scenes and render nodes carry
//! the accessibility metadata the Flow IR schema requires, that scene
//! narration meets a minimum length, and that collected themes are well
//! formed. Narration length is a style warning that escalates to an error
//! under `--strict` compilation.

import {
  diagnostic,
  hasErrors,
  type ComponentCatalog,
  type ComponentDescriptor,
  type CapabilityRegistryManifest,
  type Diagnostic,
  type JsonValue,
  type Result,
} from "@aiperf/flow-schema";

import type {
  ArgumentValueAst,
  ComponentInvocationAst,
} from "@aiperf/flow-language";

import {
  validateProps,
  type ComponentPropsSchema,
  type PropValueKind,
} from "./components.js";
import type { LinkedDocument } from "./link.js";
import { validateThemes } from "./themes.js";

/** Narration shorter than this character count is flagged as insufficient. */
const MIN_NARRATION_LENGTH = 20;

function propValueKind(type: string): PropValueKind {
  switch (type) {
    case "string":
    case "number":
    case "boolean":
      return type;
    default:
      return "json";
  }
}

function propsSchema(descriptor: ComponentDescriptor): ComponentPropsSchema {
  return {
    id: descriptor.id,
    props: Object.fromEntries(
      Object.entries(descriptor.props).map(([name, prop]) => [
        name,
        { kind: propValueKind(prop.type), required: prop.required },
      ]),
    ),
  };
}

function resolvedValue(
  value: ArgumentValueAst,
  linked: LinkedDocument,
): JsonValue {
  switch (value.kind) {
    case "literal":
      return value.value;
    case "token-reference":
      return linked.tokens.get(value.token) ?? value.token;
    case "identifier-reference":
      return value.name;
    case "object-literal":
      return Object.fromEntries(
        value.properties.map((property) => [
          property.name,
          resolvedValue(property.value, linked),
        ]),
      );
  }
}

function resolvedProps(
  invocation: ComponentInvocationAst,
  linked: LinkedDocument,
): Readonly<Record<string, JsonValue>> {
  return Object.fromEntries(
    invocation.props.map((prop) => [
      prop.name,
      resolvedValue(prop.value, linked),
    ]),
  );
}

function validateComponentInvocation(
  invocation: ComponentInvocationAst,
  linked: LinkedDocument,
  components: ComponentCatalog,
): readonly Diagnostic[] {
  const descriptor = components.components.find(
    ({ id, symbolExport }) =>
      id === invocation.name || symbolExport === invocation.name,
  );
  if (descriptor === undefined) {
    return [
      diagnostic(
        "COMPONENT_UNKNOWN",
        "error",
        `Component "${invocation.name}" is not registered.`,
        invocation.sourceMap,
        `Register a component descriptor for "${invocation.name}" or fix the invocation name.`,
      ),
    ];
  }

  return validateProps(
    resolvedProps(invocation, linked),
    propsSchema(descriptor),
    invocation.sourceMap,
  ).diagnostics;
}

/**
 * Validates capability and optional component availability, component props,
 * accessibility metadata, narration length, and linked theme declarations.
 */
export function validate(
  linked: LinkedDocument,
  capabilities: CapabilityRegistryManifest,
  strict: boolean,
  components?: ComponentCatalog,
): Result<LinkedDocument> {
  const diagnostics: Diagnostic[] = [];
  const capabilityIds = new Set(
    capabilities.capabilities.map((descriptor) => descriptor.id),
  );

  for (const requirement of linked.document.requirements) {
    if (capabilityIds.has(requirement.capability)) {
      continue;
    }
    diagnostics.push(
      diagnostic(
        "CAPABILITY_MISSING",
        "error",
        `Capability "${requirement.capability}" is not registered.`,
        requirement.sourceMap,
        `Register a capability descriptor for "${requirement.capability}" or remove the requirement.`,
      ),
    );
  }

  for (const scene of linked.document.scenes) {
    if (scene.readingOrder === undefined) {
      diagnostics.push(
        diagnostic(
          "ACCESSIBILITY_REQUIRED",
          "error",
          `Scene "${scene.id}" is missing a reading-order declaration.`,
          scene.sourceMap,
          "Add a `reading-order` declaration listing nodes in traversal order.",
        ),
      );
    }

    for (const node of scene.renderDeclarations) {
      if (node.kind === "component-invocation") {
        if (components !== undefined) {
          diagnostics.push(
            ...validateComponentInvocation(node, linked, components),
          );
        }
        continue;
      }
      if (node.label.trim().length > 0) {
        continue;
      }
      diagnostics.push(
        diagnostic(
          "ACCESSIBILITY_REQUIRED",
          "error",
          `Node "${node.id}" in scene "${scene.id}" is missing an accessible label.`,
          node.sourceMap,
          "Add a `label` declaration for this node.",
        ),
      );
    }

    const narrationText = scene.narration?.text ?? "";
    if (narrationText.trim().length < MIN_NARRATION_LENGTH) {
      diagnostics.push(
        diagnostic(
          "NARRATION_SHORT",
          strict ? "error" : "warning",
          `Scene "${scene.id}" narration is shorter than ${MIN_NARRATION_LENGTH} characters.`,
          scene.narration?.sourceMap ?? scene.sourceMap,
          "Expand the narration to describe the scene for screen-reader users.",
        ),
      );
    }
  }

  const themeResult = validateThemes({
    themes: linked.themes ?? [],
    ...(linked.useTheme === undefined ? {} : { useTheme: linked.useTheme }),
  });
  diagnostics.push(...themeResult.diagnostics);

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return { ok: true, value: linked, diagnostics };
}
