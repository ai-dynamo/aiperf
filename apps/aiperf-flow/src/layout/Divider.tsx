/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { strokeClassName } from "../theme/tokens.js";

//! Horizontal rule layout primitive; divides sections with a semantic secondary stroke.

export type DividerProps = {
  className?: string;
};

export function Divider({ className }: DividerProps): React.JSX.Element {
  return (
    <hr
      className={clsx(
        "border-t",
        "border-b-0 border-l-0 border-r-0",
        strokeClassName("secondary"),
        className,
      )}
    />
  );
}
