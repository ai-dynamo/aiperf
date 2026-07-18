/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SVGProps } from "react";
import { useHostTheme } from "../ui";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

export type MotionSignalProps = Omit<SVGProps<SVGCircleElement>, "color"> & {
  path: string;
  color?: string;
  duration?: string;
  delay?: string;
};

export function MotionSignal({
  path,
  color,
  duration = "2.2s",
  delay = "0s",
  className,
  r = 5,
  children,
  "aria-hidden": ariaHidden = true,
  ...circleProps
}: MotionSignalProps) {
  const theme = useHostTheme();
  const prefersReducedMotion = usePrefersReducedMotion();
  const motionClassName = ["motion-signal", className].filter(Boolean).join(" ");

  if (prefersReducedMotion) {
    return null;
  }

  return (
    <circle
      {...circleProps}
      r={r}
      fill={color ?? theme.category.green}
      className={motionClassName}
      aria-hidden={ariaHidden}
    >
      <animate
        attributeName="opacity"
        values="0;1;1;0"
        keyTimes="0;0.08;0.9;1"
        begin={delay}
        dur={duration}
        repeatCount="indefinite"
      />
      <animateMotion
        path={path}
        begin={delay}
        dur={duration}
        repeatCount="indefinite"
      />
      {children}
    </circle>
  );
}
