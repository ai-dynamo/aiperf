# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph utilities for manipulating radix tree structures."""

from dataclasses import asdict
from typing import Any

import numpy as np

from aiperf.dataset.synthesis.radix_tree import RadixNode, RadixTree


def merge_unary_chains(tree: RadixTree) -> None:
    """Merge unary chains (nodes with single children) into compressed edges.

    Modifies the tree in-place, combining edge labels when a node has
    only one child and that child has only one parent.

    Args:
        tree: RadixTree to compress.
    """
    # Collect all nodes to process (avoid modifying during iteration)
    nodes_to_process = [node for node in tree.get_all_nodes() if not node.is_leaf()]

    for node in nodes_to_process:
        while len(node.children) == 1:
            child_label, child = next(iter(node.children.items()))

            if child.is_leaf() or len(child.children) > 1:
                # Don't merge if child has multiple children
                break

            # Merge: remove child and update label
            if child.label is not None and child_label is not None:
                new_label = (node.label or 0) + child.label
                grandchild_label, grandchild = next(iter(child.children.items()))

                # Remove child from tree's internal tracking
                if child.node_id in tree._nodes_by_id:
                    del tree._nodes_by_id[child.node_id]

                node.children.clear()
                node.add_child(grandchild_label, grandchild)
                grandchild.label = new_label
            else:
                break


def remove_leaves(tree: RadixTree, visit_threshold: int = 1) -> None:
    """Remove leaf nodes visited only once (infrequent paths).

    Modifies the tree in-place, pruning leaves with visit count <= threshold.

    Args:
        tree: RadixTree to prune.
        visit_threshold: Minimum visit count to keep a leaf node (default: 1).
    """
    # Get all leaves before modifying
    leaves = [n for n in tree.get_all_nodes() if n.is_leaf()]

    for leaf in leaves:
        if leaf.visit_count <= visit_threshold and leaf.parent is not None:
            # Remove this leaf from parent's children
            for label, child in list(leaf.parent.children.items()):
                if child.node_id == leaf.node_id:
                    del leaf.parent.children[label]
                    # Also remove from tree's internal tracking
                    if leaf.node_id in tree._nodes_by_id:
                        del tree._nodes_by_id[leaf.node_id]
                    break


def compute_transition_cdfs(tree: RadixTree) -> dict[int, np.ndarray]:
    """Compute CDFs for outgoing transitions at each node.

    Creates cumulative distribution functions for choosing which
    child to visit based on visit counts.

    Args:
        tree: RadixTree to analyze.

    Returns:
        Dictionary mapping node IDs to CDF arrays for child transitions.
    """
    cdfs: dict[int, np.ndarray] = {}

    for node in tree.get_all_nodes():
        if node.children:
            # Count visits to each child
            visits = np.array([child.visit_count for child in node.children.values()])
            if visits.sum() > 0:
                # Compute CDF
                probs = visits / visits.sum()
                cdf = np.cumsum(probs)
                cdfs[node.node_id] = cdf

    return cdfs


def validate_tree(tree: RadixTree) -> bool:
    """Validate tree structure consistency.

    Checks that parent-child relationships are consistent
    and all nodes are reachable from root.

    Args:
        tree: RadixTree to validate.

    Returns:
        True if tree is valid, False otherwise.
    """
    # Check all non-root nodes have a parent
    for node in tree.get_all_nodes():
        if node.node_id != tree.root.node_id and node.parent is None:
            return False

    # Check all children are reachable from root
    visited = set()

    def dfs(node: RadixNode) -> None:
        visited.add(node.node_id)
        for child in node.children.values():
            if child.node_id not in visited:
                dfs(child)

    dfs(tree.root)
    return len(visited) == len(tree.get_all_nodes())


def get_tree_stats(tree: RadixTree) -> dict[str, Any]:
    """Get comprehensive statistics about tree structure.

    Args:
        tree: RadixTree to analyze.

    Returns:
        Dictionary with tree statistics.
    """
    stats = asdict(tree.get_stats())
    internal_nodes = sum(1 for n in tree.get_all_nodes() if not n.is_leaf())
    stats["internal_nodes"] = internal_nodes
    stats["branching_factor"] = sum(
        len(n.children) for n in tree.get_all_nodes()
    ) / max(internal_nodes, 1)
    return stats
