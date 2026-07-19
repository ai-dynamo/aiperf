/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ComponentDescriptor } from "../schema/component-descriptor.js";
import { diagnostic } from "../schema/diagnostic.js";
import { AIPERF_ARCHITECTURE_COMPONENTS } from "./aiperf/architecture.js";
import { AIPERF_EXECUTION_COMPONENTS } from "./aiperf/execution.js";
import { AIPERF_METRICS_COMPONENTS } from "./aiperf/metrics.js";
import { GENERIC_CHROME_COMPONENTS } from "./generic/chrome.js";
import { GENERIC_COMPOSITE_SDK_COMPONENTS } from "./generic/composites.js";
import { GENERIC_LAYOUT_SDK_COMPONENTS } from "./generic/layout.js";
import { GENERIC_MOTION_SDK_COMPONENTS } from "./generic/motion.js";
import { GENERIC_TOPOLOGY_SDK_COMPONENTS } from "./generic/topology.js";
import type {
  SdkActionName,
  SdkComponentDefinition,
  SdkComponentFactory,
} from "./types.js";

const CHROME_ACTIONS = ["enter", "emphasis", "exit"] as const satisfies readonly SdkActionName[];
const LAYOUT_ACTIONS = ["enter", "stagger"] as const satisfies readonly SdkActionName[];
const TOPOLOGY_ACTIONS = ["enter", "draw", "trace"] as const satisfies readonly SdkActionName[];
const MOTION_ACTIONS = ["enter", "pulse", "trace", "fade"] as const satisfies readonly SdkActionName[];

const CAPABILITY_BY_ID: Readonly<Record<string, string>> = {
  "sdk.header": "core.header",
  "sdk.panel": "core.panel",
  "sdk.card": "core.panel",
  "sdk.chip": "core.chip",
  "sdk.note": "core.note",
  "sdk.label": "core.text",
  "sdk.legend": "core.group",
  "sdk.callout": "core.callout",
  "sdk.divider": "core.divider",
  "sdk.bracket": "core.bracket",
  "sdk.stack": "layout.stack",
  "sdk.grid": "layout.grid",
  "sdk.rail": "layout.rail",
  "sdk.lane": "core.lane",
  "sdk.swimlane": "core.swimlane",
  "sdk.band": "core.band",
  "sdk.stepper": "core.stepper",
  "sdk.matrix": "layout.grid",
  "sdk.layerStack": "layout.stack",
  "sdk.edge": "core.connector",
  "sdk.route": "core.route",
  "sdk.pipeline": "core.group",
  "sdk.fanOut": "core.fan-out",
  "sdk.fanIn": "core.fan-in",
  "sdk.hubSpoke": "core.group",
  "sdk.tree": "core.group",
  "sdk.bidirectionalLink": "core.connector",
  "sdk.signal": "motion.signal",
  "sdk.flow": "motion.signal",
  "sdk.pulse": "motion.pulse",
  "sdk.stateTransition": "core.group",
};

const GENERIC_COMPONENT_ORDER: readonly (readonly [string, readonly SdkActionName[]])[] = [
  ["sdk.header", CHROME_ACTIONS],
  ["sdk.panel", CHROME_ACTIONS],
  ["sdk.card", CHROME_ACTIONS],
  ["sdk.chip", CHROME_ACTIONS],
  ["sdk.note", CHROME_ACTIONS],
  ["sdk.label", CHROME_ACTIONS],
  ["sdk.legend", CHROME_ACTIONS],
  ["sdk.callout", CHROME_ACTIONS],
  ["sdk.divider", CHROME_ACTIONS],
  ["sdk.bracket", CHROME_ACTIONS],
  ["sdk.stack", LAYOUT_ACTIONS],
  ["sdk.grid", LAYOUT_ACTIONS],
  ["sdk.rail", LAYOUT_ACTIONS],
  ["sdk.lane", LAYOUT_ACTIONS],
  ["sdk.swimlane", LAYOUT_ACTIONS],
  ["sdk.band", LAYOUT_ACTIONS],
  ["sdk.stepper", LAYOUT_ACTIONS],
  ["sdk.matrix", LAYOUT_ACTIONS],
  ["sdk.layerStack", LAYOUT_ACTIONS],
  ["sdk.edge", TOPOLOGY_ACTIONS],
  ["sdk.route", TOPOLOGY_ACTIONS],
  ["sdk.pipeline", TOPOLOGY_ACTIONS],
  ["sdk.fanOut", TOPOLOGY_ACTIONS],
  ["sdk.fanIn", TOPOLOGY_ACTIONS],
  ["sdk.hubSpoke", TOPOLOGY_ACTIONS],
  ["sdk.tree", TOPOLOGY_ACTIONS],
  ["sdk.bidirectionalLink", TOPOLOGY_ACTIONS],
  ["sdk.signal", MOTION_ACTIONS],
  ["sdk.flow", MOTION_ACTIONS],
  ["sdk.pulse", MOTION_ACTIONS],
  ["sdk.stateTransition", MOTION_ACTIONS],
];

function symbolExportFromId(id: string): string {
  const segment = id.includes(".") ? id.split(".", 2)[1]! : id;
  return segment.charAt(0).toUpperCase() + segment.slice(1);
}

/**
 * Normalizes an authored component id to the registry's canonical lowercase-first
 * form (e.g. `sdk.Panel` → `sdk.panel`, `sdk.FanOut` → `sdk.fanOut`).
 *
 * Native parsing requires capitalized `ComponentIdentifier` tokens after a
 * namespace qualifier; registry ids use camelCase with a lowercase initial.
 */
export function canonicalSdkComponentId(id: string): string {
  const dot = id.indexOf(".");
  if (dot < 0) {
    return id.length === 0 ? id : id.charAt(0).toLowerCase() + id.slice(1);
  }
  const namespace = id.slice(0, dot);
  const name = id.slice(dot + 1);
  if (name.length === 0) {
    return id;
  }
  return `${namespace}.${name.charAt(0).toLowerCase()}${name.slice(1)}`;
}

function createSdkDescriptor(id: string): ComponentDescriptor {
  return {
    id,
    symbolExport: symbolExportFromId(id),
    version: "1.0.0",
    classification: "flow-only",
    props: {
      id: { type: "string", required: true },
    },
    slots: {},
    events: [],
    capabilityId: CAPABILITY_BY_ID[id] ?? "core.group",
    deterministic: true,
  };
}

function createNotImplementedFactory(componentId: string): SdkComponentFactory {
  return (_props, _slots, context) => ({
    ok: false,
    diagnostics: [
      diagnostic(
        "SDK_NOT_IMPLEMENTED",
        "error",
        `SDK component "${componentId}" factory is not implemented yet.`,
        context.sourceMap,
        "This component will be available in a later SDK migration task.",
      ),
    ],
  });
}

function createStubDefinition(
  id: string,
  actions: readonly SdkActionName[],
): SdkComponentDefinition {
  return {
    descriptor: createSdkDescriptor(id),
    factory: createNotImplementedFactory(id),
    actions,
  };
}

function indexDefinitions(
  definitions: readonly SdkComponentDefinition[],
): ReadonlyMap<string, SdkComponentDefinition> {
  return new Map(definitions.map((entry) => [entry.descriptor.id, entry]));
}

const GENERIC_IMPLEMENTATIONS = indexDefinitions([
  ...GENERIC_CHROME_COMPONENTS,
  ...GENERIC_LAYOUT_SDK_COMPONENTS,
  ...GENERIC_TOPOLOGY_SDK_COMPONENTS,
  ...GENERIC_MOTION_SDK_COMPONENTS,
  ...GENERIC_COMPOSITE_SDK_COMPONENTS,
]);

/** Generic SDK pack component definitions. */
export const GENERIC_SDK_COMPONENTS: readonly SdkComponentDefinition[] =
  GENERIC_COMPONENT_ORDER.map(([id, actions]) => {
    const implemented = GENERIC_IMPLEMENTATIONS.get(id);
    return implemented ?? createStubDefinition(id, actions);
  });

/** AIPerf SDK pack component definitions. */
export const AIPERF_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  ...AIPERF_ARCHITECTURE_COMPONENTS,
  ...AIPERF_EXECUTION_COMPONENTS,
  ...AIPERF_METRICS_COMPONENTS,
];

export type SdkRegistry = Readonly<{
  components: readonly SdkComponentDefinition[];
  lookup: (id: string) => SdkComponentDefinition | undefined;
}>;

function buildRegistry(
  components: readonly SdkComponentDefinition[],
): SdkRegistry {
  const byId = new Map(components.map((entry) => [entry.descriptor.id, entry]));
  return {
    components,
    lookup: (id: string) =>
      byId.get(id) ?? byId.get(canonicalSdkComponentId(id)),
  };
}

const DEFAULT_SDK_REGISTRY = buildRegistry([
  ...GENERIC_SDK_COMPONENTS,
  ...AIPERF_SDK_COMPONENTS,
]);

/** Returns the typed SDK component registry for compile-time expansion. */
export function createSdkRegistry(): SdkRegistry {
  return DEFAULT_SDK_REGISTRY;
}

/** Looks up an SDK component definition by canonical id (e.g. `sdk.panel`). */
export function lookupSdkComponent(id: string): SdkComponentDefinition | undefined {
  return DEFAULT_SDK_REGISTRY.lookup(id);
}
