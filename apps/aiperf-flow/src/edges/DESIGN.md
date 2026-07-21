# FlowEdge design note

`FlowEdge` is a custom React Flow edge type (`type: "flow"`) for signaling data
movement along a connection — the animated dashed "flow line" used throughout
the source diagrams to show request/data flow through a pipeline.

Approach: compute the edge geometry with `getBezierPath` from `@xyflow/react`
(same primitive React Flow's own default/smoothstep edges use), render a single
`<path>` with a dashed `strokeDasharray`, and animate the dash phase with a CSS
`@keyframes` rule that decrements `stroke-dashoffset` continuously. Decreasing
offset moves the dashes in the direction of the path (source → target), which
reads as flow. Duration is driven by `data.speed` (`slow` | `normal` | `fast`,
default `normal`) mapped to a fixed set of durations; stroke color comes from
`data.color`, defaulting to `"var(--color-accent-primary)"` — a CSS variable
reference is used instead of a Tailwind class name because raw hex/var values
are what an SVG `stroke` attribute needs, and this keeps the component token-driven
without inventing a new JS-side color-role-to-hex map.

`prefers-reduced-motion` is handled with a CSS media query
(`@media (prefers-reduced-motion: reduce)`) that sets `animation: none` on the
dash class, rather than a JS `matchMedia` check — simpler, no extra render
path, and it stays correct even if the OS setting changes while mounted.
