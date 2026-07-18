/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic, content-addressed packing of validated Flow IR.

import type {
  CapabilityRequirement,
  FlowIr,
  SceneIr,
} from "../schema/index.js";
import { sha256 } from "js-sha256";

/** A single packed artifact with its content-addressed hash. */
export type PackedFile = Readonly<{
  path: string;
  content: Uint8Array;
  mediaType: string;
  hash: string;
}>;

/** Top-level manifest describing every artifact in a packed Flow bundle. */
export type PackManifest = Readonly<{
  formatVersion: 1;
  id: string;
  title: string;
  sourceName: string;
  capabilities: readonly CapabilityRequirement[];
  scenes: readonly {
    id: string;
    title: string;
    chunkPath: string;
    hash: string;
  }[];
  transcriptPath: string;
  contentHash: string;
}>;

/** A packed Flow bundle: its manifest and every file it references. */
export type PackedFlow = Readonly<{
  manifest: PackManifest;
  files: readonly PackedFile[];
}>;

function normalize(value: unknown): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(normalize);
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, normalize(entry)]),
  );
}

/** Serializes a value as deterministic, key-sorted JSON bytes. */
export function canonicalJson(value: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(normalize(value)));
}

function hashBytes(bytes: Uint8Array): string {
  return sha256(bytes);
}

function sceneChunkPath(scene: SceneIr): string {
  return `chunks/scene-${scene.id}.json`;
}

function transcriptText(ir: FlowIr): string {
  return ir.scenes
    .map((scene) => `${scene.title}\n${scene.narration}`.trim())
    .join("\n\n");
}

/** Packs validated Flow IR into a deterministic, content-addressed bundle. */
export function packFlow(ir: FlowIr, sourceName: string): PackedFlow {
  const sceneEntries = ir.scenes.map((scene) => {
    const content = canonicalJson(scene);
    const hash = hashBytes(content);
    const file: PackedFile = {
      path: sceneChunkPath(scene),
      content,
      mediaType: "application/json",
      hash,
    };
    return {
      file,
      manifestEntry: {
        id: scene.id,
        title: scene.title,
        chunkPath: file.path,
        hash,
      },
    };
  });

  const transcriptContent = new TextEncoder().encode(transcriptText(ir));
  const transcriptFile: PackedFile = {
    path: "transcript.txt",
    content: transcriptContent,
    mediaType: "text/plain",
    hash: hashBytes(transcriptContent),
  };

  const contentAddressedFiles = [
    ...sceneEntries.map((entry) => entry.file),
    transcriptFile,
  ];
  const contentHash = hashBytes(
    canonicalJson(
      contentAddressedFiles.map(({ path, hash }) => ({ path, hash })),
    ),
  );

  const manifest: PackManifest = {
    formatVersion: 1,
    id: ir.id,
    title: ir.title,
    sourceName,
    capabilities: ir.capabilities,
    scenes: sceneEntries.map((entry) => entry.manifestEntry),
    transcriptPath: transcriptFile.path,
    contentHash,
  };

  const manifestContent = canonicalJson(manifest);
  const manifestFile: PackedFile = {
    path: "flow.manifest.json",
    content: manifestContent,
    mediaType: "application/json",
    hash: hashBytes(manifestContent),
  };

  const files = [...contentAddressedFiles, manifestFile].sort((left, right) =>
    left.path.localeCompare(right.path),
  );

  return { manifest, files };
}
