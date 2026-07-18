// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";

import { FlowApp } from "./app.js";
import { createFoundationRegistry } from "./renderer.js";
import type { CapabilityRegistry } from "./registry.js";
import "./theme.css";

type UnknownRecord = Readonly<Record<string, unknown>>;
type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function sourceMap(source: string): UnknownRecord {
  const position = { offset: 0, line: 1, column: 1 };
  return { source, start: position, end: position };
}

async function fetchJson(fetcher: Fetcher, path: string): Promise<unknown> {
  const response = await fetcher(path);
  if (!response.ok) {
    throw new Error(`Unable to load ${path}: HTTP ${response.status}.`);
  }
  return response.json();
}

function chunkUrl(path: string): string {
  if (
    path === "" ||
    path.startsWith("/") ||
    path.includes("..") ||
    /^[a-z][a-z\d+.-]*:/i.test(path)
  ) {
    throw new Error(`Invalid scene chunk path "${path}".`);
  }
  return `./${path.replace(/^\.\//, "")}`;
}

function sceneFallback(descriptor: UnknownRecord, source: string): SceneIr {
  return {
    id: text(descriptor.id, "unavailable-scene"),
    title: text(descriptor.title, "Unavailable scene"),
    summary: text(descriptor.summary, "This scene could not be loaded."),
    roots: null,
    camera: [],
    timeline: [],
    narration: text(descriptor.transcript),
    interactions: [],
    responsive: [],
    accessibility: {
      label: text(descriptor.title, "Unavailable scene"),
      readingOrder: [],
    },
    fallback: text(descriptor.fallback, "Use the transcript for this scene."),
    sourceMap: sourceMap(source),
  } as unknown as SceneIr;
}

export async function loadPackedFlow(
  fetcher: Fetcher = fetch,
  registry: CapabilityRegistry = createFoundationRegistry(),
): Promise<FlowIr> {
  const manifest = record(await fetchJson(fetcher, "./flow.manifest.json"));
  const formatVersion = manifest.formatVersion ?? manifest.version;
  if (formatVersion !== 1) {
    throw new Error(
      `Unsupported Flow manifest version ${String(formatVersion)}; expected 1.`,
    );
  }

  const requirementsValue =
    manifest.requiredCapabilities ?? manifest.capabilities;
  const requirements = Array.isArray(requirementsValue)
    ? requirementsValue
    : [];
  for (const requirement of requirements) {
    registry.require(text(record(requirement).id));
  }

  const descriptorsValue = manifest.scenes;
  if (!Array.isArray(descriptorsValue) || descriptorsValue.length === 0) {
    throw new Error("Flow manifest does not contain any scenes.");
  }
  const source = text(
    manifest.source ?? manifest.sourceFilename,
    "flow.manifest.json",
  );
  const descriptors = descriptorsValue.map(record);
  const scenes = descriptors.map((descriptor) =>
    sceneFallback(descriptor, source),
  );
  const firstDescriptor = descriptors[0] ?? {};
  const firstChunkPath = text(
    firstDescriptor.chunkPath ??
      firstDescriptor.chunk ??
      firstDescriptor.path,
  );
  try {
    scenes[0] = (await fetchJson(
      fetcher,
      chunkUrl(firstChunkPath),
    )) as SceneIr;
  } catch {
    scenes[0] = sceneFallback(firstDescriptor, source);
  }

  const metadata = record(manifest.flow);
  return {
    irVersion: 1,
    id: text(manifest.id ?? metadata.id, "packed-flow"),
    title: text(manifest.title ?? metadata.title, "AIPerf Flow"),
    capabilities: requirements,
    tokens: record(manifest.tokens),
    scenes,
    sourceMap: sourceMap(source),
  } as unknown as FlowIr;
}

function SiteLoadFailure({ error }: Readonly<{ error: Error }>): ReactNode {
  return (
    <main className="aiperf-flow aiperf-flow__site-error">
      <h1>Flow could not be loaded</h1>
      <p>{error.message}</p>
      <p>The manifest and scene files must be served from the same static site.</p>
    </main>
  );
}

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Loads the packed Flow site and mounts {@link FlowApp}. The app evaluates
 * scene IR into a display list, paints Canvas when available (else SVG), and
 * always mounts the semantic HTML twin beside the visual stage.
 */
export async function mountFlowSite(
  rootElement: HTMLElement,
  fetcher: Fetcher = fetch,
): Promise<void> {
  const root = createRoot(rootElement);
  try {
    root.render(
      <FlowApp
        flow={await loadPackedFlow(fetcher)}
        reducedMotion={prefersReducedMotion()}
      />,
    );
  } catch (error) {
    root.render(
      <SiteLoadFailure
        error={
          error instanceof Error ? error : new Error("Unknown site load error.")
        }
      />,
    );
  }
}

const rootElement = document.getElementById("root");
if (rootElement !== null) {
  void mountFlowSite(rootElement);
}
