/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";
import { strokeClassName, surfaceClassName } from "../theme/tokens.js";
import type { SurfaceRole } from "../theme/tokens.js";

export type FramedProps = {
  children: ReactNode;
  /** Surface role for the panel background. Defaults to `"page"`. */
  surfaceRole?: SurfaceRole;
  className?: string;
};

/**
 * A soft-bordered content panel used to group prose without the weight of a full `Callout` or
 * diagram `Card`. Consolidates the local `Framed` one-off built for this shape.
 */
export function Framed({ children, surfaceRole = "page", className }: FramedProps): React.JSX.Element {
  return (
    <div className={clsx("rounded-none border p-3", strokeClassName("tertiary"), surfaceClassName(surfaceRole), className)}>
      {children}
    </div>
  );
}
