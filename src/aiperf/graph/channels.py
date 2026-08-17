# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-engine channel primitives for the async-dataflow graph engine.

Single source of truth for static channel topology derivations shared by the
executor (``"all"`` fan-in resolution in the channel store) and the static
trace analyzer. The only derivation here today is ``producers_per_channel``;
the dataflow channel store and the analyzer both consume it instead of
re-deriving per-channel producer counts at their own callsites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import GraphRecord

__all__ = ["producers_per_channel"]


def producers_per_channel(graph: GraphRecord) -> dict[str, int]:
    """Count the nodes that statically write each channel.

    A channel's producer count is the number of nodes whose ``write_channels``
    (the property on each node of the typed graph model) include that channel. In
    the flat ``LlmNode`` form the only
    writer is ``LlmNode.output``; ``write_channels`` covers it uniformly, so no
    per-kind special-casing is needed here.

    Every channel declared in ``graph.state`` is seeded to ``0`` so the result
    carries an entry for declared-but-unwritten channels (e.g. pure initial
    state). Counts for channels a node writes that are not in ``graph.state``
    (such as error markers) are still included.

    Args:
        graph: The static agent graph whose nodes' ``write_channels`` are counted.

    Returns:
        Mapping of channel name to the number of nodes that write it.
    """
    counts: dict[str, int] = {ch: 0 for ch in graph.state}
    for node in graph.nodes.values():
        for ch in node.write_channels:
            counts[ch] = counts.get(ch, 0) + 1
    return counts
