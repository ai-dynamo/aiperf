# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Radix tree data structure for representing prefix relationships."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class RadixTreeStats:
    """Statistics about radix tree structure.

    Attributes:
        num_nodes: Total number of nodes in tree.
        num_leaves: Number of leaf nodes (nodes with no children).
        total_visits: Number of paths added to tree.
        max_depth: Maximum depth from root to leaf.
    """

    num_nodes: int
    num_leaves: int
    total_visits: int
    max_depth: int


@dataclass(slots=True)
class RadixNode:
    """A node in the radix tree representing a prefix path.

    Attributes:
        node_id: Unique node identifier.
        label: Edge label (token count).
        visit_count: Number of times this node is visited.
        children: Child nodes by edge label.
        parent: Parent node reference.
    """

    node_id: int
    label: int | None = None
    visit_count: int = 0
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    parent: "RadixNode | None" = None

    def add_child(self, label: int, child: "RadixNode") -> None:
        """Add a child node with the given edge label.

        Args:
            label: Edge label (typically hash ID or token count).
            child: Child RadixNode to add.
        """
        self.children[label] = child
        child.parent = self

    def get_child(self, label: int) -> "RadixNode | None":
        """Get child with given label, or None if not present.

        Args:
            label: Edge label to look up.

        Returns:
            Child RadixNode or None.
        """
        return self.children.get(label)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children).

        Returns:
            True if node has no children.
        """
        return len(self.children) == 0


class RadixTree:
    """A radix tree for compactly representing prefix patterns.

    The radix tree compresses unary chains (nodes with single child)
    into edges with larger labels, representing token counts.
    """

    def __init__(self) -> None:
        """Initialize an empty radix tree."""
        self._root = RadixNode(node_id=0, label=None)
        self._node_id_counter = 1
        self._nodes_by_id: dict[int, RadixNode] = {0: self._root}

    @property
    def root(self) -> RadixNode:
        """Get the root node of the tree.

        Returns:
            The root RadixNode.
        """
        return self._root

    def add_path(self, path: list[int]) -> RadixNode:
        """Add a path to the tree from root, creating nodes as needed.

        Args:
            path: List of edge labels (hash IDs or token counts) representing a path.

        Returns:
            The leaf node at the end of the path.
        """
        current = self._root
        current.visit_count += 1
        for label in path:
            child = current.get_child(label)
            if child is None:
                child = self._create_node(label)
                current.add_child(label, child)
            current = child
            current.visit_count += 1

        return current

    def _create_node(self, label: int | None) -> RadixNode:
        """Create a new node.

        Args:
            label: Optional edge label.

        Returns:
            New RadixNode.
        """
        node = RadixNode(node_id=self._node_id_counter, label=label)
        self._node_id_counter += 1
        self._nodes_by_id[node.node_id] = node
        return node

    def get_node(self, node_id: int) -> RadixNode | None:
        """Get node by ID.

        Args:
            node_id: Node identifier.

        Returns:
            RadixNode or None if not found.
        """
        return self._nodes_by_id.get(node_id)

    def get_all_nodes(self) -> list[RadixNode]:
        """Get all nodes in the tree.

        Returns:
            List of all RadixNode instances.
        """
        return list(self._nodes_by_id.values())

    def get_stats(self) -> RadixTreeStats:
        """Get statistics about the tree structure.

        Returns:
            RadixTreeStats with tree statistics.
        """
        return RadixTreeStats(
            num_nodes=len(self._nodes_by_id),
            num_leaves=sum(1 for node in self._nodes_by_id.values() if node.is_leaf()),
            total_visits=self._root.visit_count,
            max_depth=self._compute_max_depth(),
        )

    def _compute_max_depth(self) -> int:
        """Compute the maximum depth of the tree from root.

        Returns:
            Maximum depth (distance from root to farthest leaf).
        """

        def dfs(node: RadixNode) -> int:
            if node.is_leaf():
                return 1
            return 1 + max((dfs(child) for child in node.children.values()), default=0)

        return dfs(self._root) - 1  # Subtract 1 to not count root
