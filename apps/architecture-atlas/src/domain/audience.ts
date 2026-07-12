// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from "zod";

export const audienceSchema = z.enum([
  "executive",
  "developer",
  "maintainer",
]);

export type Audience = z.infer<typeof audienceSchema>;

export const DEFAULT_AUDIENCE: Audience = "developer";
export const AUDIENCE_STORAGE_KEY = "aiperf-atlas:audience";

export const audienceSearchSchema = z.object({
  audience: audienceSchema.optional(),
});

export type AudienceSearch = z.infer<typeof audienceSearchSchema>;

interface AudienceStorage {
  getItem(key: string): string | null;
}

interface WritableAudienceStorage extends AudienceStorage {
  setItem(key: string, value: string): void;
}

export function parseAudienceSearch(
  search: Record<string, unknown>,
): AudienceSearch {
  const result = audienceSearchSchema.safeParse(search);
  return result.success ? result.data : {};
}

export function readStoredAudience(storage: AudienceStorage): Audience {
  try {
    const result = audienceSchema.safeParse(
      storage.getItem(AUDIENCE_STORAGE_KEY),
    );
    return result.success ? result.data : DEFAULT_AUDIENCE;
  } catch {
    return DEFAULT_AUDIENCE;
  }
}

export function resolveAudience(
  urlAudience: unknown,
  storage: AudienceStorage,
): Audience {
  const result = audienceSchema.safeParse(urlAudience);
  return result.success ? result.data : readStoredAudience(storage);
}

export function persistAudience(
  audience: Audience,
  storage: WritableAudienceStorage,
): void {
  try {
    storage.setItem(AUDIENCE_STORAGE_KEY, audience);
  } catch {
    // Browser privacy settings may deny storage while URL state remains usable.
  }
}
