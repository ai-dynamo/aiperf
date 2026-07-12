// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvidenceReference } from "../../domain/architecture";

export const REPOSITORY_SOURCE_BASE_URL =
  import.meta.env.VITE_ARCHITECTURE_SOURCE_BASE_URL?.trim() ||
  "https://github.com/ai-dynamo/aiperf/blob/main";

interface RepositorySource {
  label: string;
  href?: string;
}

export function repositorySource(
  evidence: EvidenceReference,
  webBase: string | undefined,
): RepositorySource {
  const suffix = evidence.lines
    ? `:${evidence.lines.start}-${evidence.lines.end}`
    : evidence.symbol
      ? `#${evidence.symbol}`
      : "";
  const source: RepositorySource = {
    label: `${evidence.path}${suffix}`,
  };
  if (!webBase?.trim()) {
    return source;
  }
  const fragment = evidence.lines
    ? `#L${evidence.lines.start}-L${evidence.lines.end}`
    : "";
  return {
    ...source,
    href: `${webBase.replace(/\/+$/u, "")}/${evidence.path}${fragment}`,
  };
}
