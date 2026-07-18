// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext, type ReactNode } from "react";

import type { Audience } from "../domain/audience";

const AudienceContext = createContext<Audience | null>(null);

interface AudienceProviderProps {
  audience: Audience;
  children: ReactNode;
}

export function AudienceProvider({
  audience,
  children,
}: AudienceProviderProps) {
  return (
    <AudienceContext.Provider value={audience}>
      {children}
    </AudienceContext.Provider>
  );
}

export function useAudience(): Audience {
  const audience = useContext(AudienceContext);
  if (!audience) {
    throw new Error("Guided architecture views require an audience provider.");
  }
  return audience;
}
