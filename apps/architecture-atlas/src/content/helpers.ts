// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AudienceCopy,
  EvidenceReference,
} from "../domain/architecture";

export function copy(
  executive: string,
  developer: string,
  maintainer: string,
): AudienceCopy {
  return { executive, developer, maintainer };
}

export function evidence(
  path: string,
  symbol?: string,
): EvidenceReference {
  return symbol ? { path, symbol } : { path };
}

export function rangedEvidence(
  path: string,
  start: number,
  end: number,
  symbol?: string,
): EvidenceReference {
  return {
    path,
    lines: { start, end },
    ...(symbol ? { symbol } : {}),
  };
}
