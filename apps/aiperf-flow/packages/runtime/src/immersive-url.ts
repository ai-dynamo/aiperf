// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Serializable immersive selection restored from the URL query string. */
export type ImmersiveUrlState = Readonly<{
  sceneId: string | null;
  beatId: string | null;
  entityId: string | null;
}>;

const QUERY_KEYS = {
  sceneId: "scene",
  beatId: "beat",
  entityId: "entity",
} as const;

/** Parses decoded immersive selections without reading browser state. */
export function parseImmersiveUrl(search: string): ImmersiveUrlState {
  const parameters = new URLSearchParams(search);
  return {
    sceneId: parameters.get(QUERY_KEYS.sceneId),
    beatId: parameters.get(QUERY_KEYS.beatId),
    entityId: parameters.get(QUERY_KEYS.entityId),
  };
}

/** Serializes immersive selections in stable scene, beat, entity order. */
export function serializeImmersiveUrl(state: ImmersiveUrlState): string {
  const parameters = new URLSearchParams();
  for (const key of Object.keys(QUERY_KEYS) as (keyof ImmersiveUrlState)[]) {
    const value = state[key];
    if (value !== null) {
      parameters.set(QUERY_KEYS[key], value);
    }
  }

  const search = parameters.toString();
  return search === "" ? "" : `?${search}`;
}
