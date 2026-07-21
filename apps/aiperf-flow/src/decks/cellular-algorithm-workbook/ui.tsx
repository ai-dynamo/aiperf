/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small local status-chip helpers scoped to the cellular-algorithm-workbook deck, built on the
//! shared `Pill`/`Eyebrow`/`Framed` prose primitives.

import type { Status } from "./data.js";

export { Pill } from "../../prose/Pill.js";
export { Eyebrow } from "../../prose/Eyebrow.js";
export { Framed } from "../../prose/Framed.js";

import { Pill } from "../../prose/Pill.js";

const STATUS_LABELS: Readonly<Record<Status, string>> = {
  built: "Built",
  partial: "Partial",
  "feature-gated": "Feature gated",
  approximate: "Approximate",
  rejected: "Rejected",
};

/** Implementation-status chip; `rejected` is drawn in the danger category, others neutral. */
export function StatusLabel({ status }: { status: Status }): React.JSX.Element {
  const rejected = status === "rejected";
  return (
    <Pill tone={rejected ? "red" : undefined} ariaLabel={`Implementation status: ${STATUS_LABELS[status]}`}>
      {STATUS_LABELS[status]}
    </Pill>
  );
}

/** Route admission chip: Admitted (neutral) vs Rejected (danger). */
export function AdmissionLabel({ valid }: { valid: boolean }): React.JSX.Element {
  return (
    <Pill tone={valid ? undefined : "red"} ariaLabel={`Route admission: ${valid ? "Admitted" : "Rejected"}`}>
      {valid ? "Admitted" : "Rejected"}
    </Pill>
  );
}
