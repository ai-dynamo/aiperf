// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

interface RoutePlaceholderProps {
  eyebrow: string;
  title: string;
  summary: string;
}

export function RoutePlaceholder({
  eyebrow,
  title,
  summary,
}: RoutePlaceholderProps) {
  return (
    <section className="route-stage" aria-labelledby="route-title">
      <p className="route-eyebrow">{eyebrow}</p>
      <h1 id="route-title">{title}</h1>
      <p className="route-summary">{summary}</p>
      <div className="handoff-rail" aria-label="Product handoff sequence">
        <span>Python authoring</span>
        <span>Runner boundary</span>
        <span>Execution seams</span>
        <span>Native report</span>
        <span>Python presentation</span>
      </div>
      <p className="foundation-note">
        Architecture evidence and interactive diagrams are introduced in the
        next content milestone.
      </p>
    </section>
  );
}
