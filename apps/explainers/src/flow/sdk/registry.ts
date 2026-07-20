/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ComponentDescriptor } from "../schema/component-descriptor.js";
import { diagnostic } from "../schema/diagnostic.js";
import { AIPERF_ARCHITECTURE_COMPONENTS } from "./aiperf/architecture.js";
import { AIPERF_EXECUTION_COMPONENTS } from "./aiperf/execution.js";
import { AIPERF_METRICS_COMPONENTS } from "./aiperf/metrics.js";
import { DIAGRAM_SDK_COMPONENTS } from "./diagram/catalog.js";
import { GENERIC_CATALOG_COMPONENTS } from "./generic/catalog.js";
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
  "sdk.shape": "core.rect",
  "sdk.text": "core.text",
  "sdk.richText": "core.text",
  "sdk.icon": "core.path",
  "sdk.image": "core.image",
  "sdk.line": "core.line",
  "sdk.arrow": "core.arrow",
  "sdk.spacer": "core.group",
  "sdk.inset": "layout.pad",
  "sdk.title": "core.text",
  "sdk.paragraph": "core.text",
  "sdk.caption": "core.text",
  "sdk.codeBlock": "core.group",
  "sdk.quote": "core.group",
  "sdk.list": "core.group",
  "sdk.keyValue": "core.group",
  "sdk.propertyList": "core.group",
  "sdk.badge": "core.chip",
  "sdk.statusDot": "core.circle",
  "sdk.avatar": "core.group",
  "sdk.iconLabel": "core.group",
  "sdk.alert": "core.panel",
  "sdk.statusCard": "core.panel",
  "sdk.emptyState": "core.panel",
  "sdk.stat": "core.group",
  "sdk.metric": "core.group",
  "sdk.table": "core.group",
  "sdk.tableRow": "layout.rail",
  "sdk.tableCell": "core.group",
  "sdk.tagList": "layout.rail",
  "sdk.breadcrumb": "layout.rail",
  "sdk.tabs": "layout.rail",
  "sdk.pagination": "layout.rail",
  "sdk.timeline": "core.group",
  "sdk.timelineItem": "core.group",
  "sdk.progress": "core.group",
  "sdk.meter": "core.group",
  "sdk.gauge": "core.group",
  "sdk.sparkline": "core.group",
  "sdk.rating": "layout.rail",
  "sdk.semaphore": "layout.rail",
  "sdk.section": "core.group",
  "sdk.toolbar": "layout.rail",
  "sdk.splitPane": "layout.stack",
  "sdk.mediaObject": "core.group",
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
  "sdk.overlay": "layout.overlay",
  "sdk.frame": "layout.frame",
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
  "sdk.user": "diagram.actor",
  "sdk.client": "diagram.compute",
  "sdk.service": "diagram.compute",
  "sdk.server": "diagram.compute",
  "sdk.process": "diagram.compute",
  "sdk.worker": "diagram.compute",
  "sdk.function": "diagram.compute",
  "sdk.container": "diagram.compute",
  "sdk.cloud": "diagram.cloud",
  "sdk.database": "diagram.storage",
  "sdk.dataStore": "diagram.storage",
  "sdk.cache": "diagram.storage",
  "sdk.file": "diagram.storage",
  "sdk.objectStore": "diagram.storage",
  "sdk.volume": "diagram.storage",
  "sdk.queue": "diagram.messaging",
  "sdk.topic": "diagram.messaging",
  "sdk.stream": "diagram.messaging",
  "sdk.eventBus": "diagram.messaging",
  "sdk.gateway": "diagram.network",
  "sdk.endpoint": "diagram.network",
  "sdk.loadBalancer": "diagram.network",
  "sdk.firewall": "diagram.network",
  "sdk.start": "diagram.control",
  "sdk.end": "diagram.control",
  "sdk.processStep": "diagram.control",
  "sdk.decision": "diagram.control",
  "sdk.merge": "diagram.control",
  "sdk.delay": "diagram.control",
  "sdk.retry": "diagram.control",
  "sdk.loop": "diagram.control",
  "sdk.boundary": "diagram.boundary",
  "sdk.zone": "diagram.boundary",
  "sdk.cluster": "diagram.boundary",
  "sdk.trustBoundary": "diagram.boundary",
  "sdk.document": "diagram.symbol",
  "sdk.terminal": "diagram.symbol",
  "sdk.clock": "diagram.symbol",
  "sdk.lock": "diagram.symbol",
  "sdk.key": "diagram.symbol",
  "sdk.warning": "diagram.symbol",
};

const GENERIC_COMPONENT_ORDER: readonly (readonly [string, readonly SdkActionName[]])[] = [
  ["sdk.shape", CHROME_ACTIONS],
  ["sdk.text", CHROME_ACTIONS],
  ["sdk.richText", CHROME_ACTIONS],
  ["sdk.icon", CHROME_ACTIONS],
  ["sdk.image", CHROME_ACTIONS],
  ["sdk.line", TOPOLOGY_ACTIONS],
  ["sdk.arrow", TOPOLOGY_ACTIONS],
  ["sdk.spacer", LAYOUT_ACTIONS],
  ["sdk.inset", LAYOUT_ACTIONS],
  ["sdk.title", CHROME_ACTIONS],
  ["sdk.paragraph", CHROME_ACTIONS],
  ["sdk.caption", CHROME_ACTIONS],
  ["sdk.codeBlock", CHROME_ACTIONS],
  ["sdk.quote", CHROME_ACTIONS],
  ["sdk.list", LAYOUT_ACTIONS],
  ["sdk.keyValue", CHROME_ACTIONS],
  ["sdk.propertyList", LAYOUT_ACTIONS],
  ["sdk.badge", CHROME_ACTIONS],
  ["sdk.statusDot", CHROME_ACTIONS],
  ["sdk.avatar", CHROME_ACTIONS],
  ["sdk.iconLabel", CHROME_ACTIONS],
  ["sdk.alert", CHROME_ACTIONS],
  ["sdk.statusCard", CHROME_ACTIONS],
  ["sdk.emptyState", CHROME_ACTIONS],
  ["sdk.stat", CHROME_ACTIONS],
  ["sdk.metric", CHROME_ACTIONS],
  ["sdk.table", LAYOUT_ACTIONS],
  ["sdk.tableRow", LAYOUT_ACTIONS],
  ["sdk.tableCell", CHROME_ACTIONS],
  ["sdk.tagList", LAYOUT_ACTIONS],
  ["sdk.breadcrumb", LAYOUT_ACTIONS],
  ["sdk.tabs", LAYOUT_ACTIONS],
  ["sdk.pagination", LAYOUT_ACTIONS],
  ["sdk.timeline", LAYOUT_ACTIONS],
  ["sdk.timelineItem", CHROME_ACTIONS],
  ["sdk.progress", MOTION_ACTIONS],
  ["sdk.meter", MOTION_ACTIONS],
  ["sdk.gauge", MOTION_ACTIONS],
  ["sdk.sparkline", MOTION_ACTIONS],
  ["sdk.rating", CHROME_ACTIONS],
  ["sdk.semaphore", MOTION_ACTIONS],
  ["sdk.section", LAYOUT_ACTIONS],
  ["sdk.toolbar", LAYOUT_ACTIONS],
  ["sdk.splitPane", LAYOUT_ACTIONS],
  ["sdk.mediaObject", LAYOUT_ACTIONS],
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
  ["sdk.overlay", LAYOUT_ACTIONS],
  ["sdk.frame", LAYOUT_ACTIONS],
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
  ["sdk.user", CHROME_ACTIONS],
  ["sdk.client", CHROME_ACTIONS],
  ["sdk.service", CHROME_ACTIONS],
  ["sdk.server", CHROME_ACTIONS],
  ["sdk.process", CHROME_ACTIONS],
  ["sdk.worker", CHROME_ACTIONS],
  ["sdk.function", CHROME_ACTIONS],
  ["sdk.container", CHROME_ACTIONS],
  ["sdk.cloud", CHROME_ACTIONS],
  ["sdk.database", CHROME_ACTIONS],
  ["sdk.dataStore", CHROME_ACTIONS],
  ["sdk.cache", CHROME_ACTIONS],
  ["sdk.file", CHROME_ACTIONS],
  ["sdk.objectStore", CHROME_ACTIONS],
  ["sdk.volume", CHROME_ACTIONS],
  ["sdk.queue", TOPOLOGY_ACTIONS],
  ["sdk.topic", TOPOLOGY_ACTIONS],
  ["sdk.stream", TOPOLOGY_ACTIONS],
  ["sdk.eventBus", TOPOLOGY_ACTIONS],
  ["sdk.gateway", TOPOLOGY_ACTIONS],
  ["sdk.endpoint", TOPOLOGY_ACTIONS],
  ["sdk.loadBalancer", TOPOLOGY_ACTIONS],
  ["sdk.firewall", TOPOLOGY_ACTIONS],
  ["sdk.start", TOPOLOGY_ACTIONS],
  ["sdk.end", TOPOLOGY_ACTIONS],
  ["sdk.processStep", TOPOLOGY_ACTIONS],
  ["sdk.decision", TOPOLOGY_ACTIONS],
  ["sdk.merge", TOPOLOGY_ACTIONS],
  ["sdk.delay", TOPOLOGY_ACTIONS],
  ["sdk.retry", TOPOLOGY_ACTIONS],
  ["sdk.loop", TOPOLOGY_ACTIONS],
  ["sdk.boundary", LAYOUT_ACTIONS],
  ["sdk.zone", LAYOUT_ACTIONS],
  ["sdk.cluster", LAYOUT_ACTIONS],
  ["sdk.trustBoundary", LAYOUT_ACTIONS],
  ["sdk.document", CHROME_ACTIONS],
  ["sdk.terminal", CHROME_ACTIONS],
  ["sdk.clock", CHROME_ACTIONS],
  ["sdk.lock", CHROME_ACTIONS],
  ["sdk.key", CHROME_ACTIONS],
  ["sdk.warning", CHROME_ACTIONS],
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
  ...GENERIC_CATALOG_COMPONENTS,
  ...GENERIC_CHROME_COMPONENTS,
  ...GENERIC_LAYOUT_SDK_COMPONENTS,
  ...GENERIC_TOPOLOGY_SDK_COMPONENTS,
  ...GENERIC_MOTION_SDK_COMPONENTS,
  ...GENERIC_COMPOSITE_SDK_COMPONENTS,
  ...DIAGRAM_SDK_COMPONENTS,
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
