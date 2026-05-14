---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Code Patterns
---
# AIPerf Code Patterns

Code examples for common development tasks. Referenced from CLAUDE.md.

Each pattern lives in its own page below. Open the one that matches the surface you're working on, copy the snippet, and adapt the names.

## Service surface

- [CLI Command Pattern](./cli-command.md) — adding a new `aiperf <command>` (lazy registration, cyclopts shape).
- [Service Pattern](./service.md) — implementing a new `BaseComponentService` with lifecycle hooks.
- [Plugin System Pattern](./plugin-system.md) — registering a plugin in `plugins.yaml` and consuming via `plugins.get_class(...)`.
- [Strategy Protocol Pattern](./strategy-protocol.md) — adding a new strategy that other code dispatches over.

## Data surface

- [Model Pattern](./model.md) — `AIPerfBaseModel` for data, `BaseConfig` for configuration, `@dataclass(slots=True)` for hot-path dataclasses.
- [Message Pattern](./message.md) — defining a `Message` subclass and the `@on_message` handler on the receiving side.

## Runtime / infrastructure surface

- [Error Handling Pattern](./error-handling.md) — informative exception messages and `raise ... from e` discipline.
- [Logging Pattern](./logging.md) — lambda for expensive log message construction, direct strings for cheap ones.
- [Console Exporter Pattern](./console-exporter.md) — gated single-table exporters and the `MetricConsoleGroup` flag.
- [Drop-Oldest Fanout Queue](./drop-oldest-fanout-queue.md) — backpressure-tolerant fan-out queue for streaming subscribers.

## Visualization

- [Uncertainty Plot Pattern](./uncertainty-plot.md) — the data contract, Plotly + Matplotlib renderers, and the ellipse-geometry utility.

## Testing

- [Testing Pattern](./testing.md) — pytest markers, fixtures, the `parametrize` + `# fmt: skip` convention, and the harness modules.
