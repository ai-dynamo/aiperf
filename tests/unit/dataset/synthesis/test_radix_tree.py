# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RadixTree and RadixNode."""

import pytest

from aiperf.dataset.synthesis import RadixNode, RadixTree, RadixTreeStats


class TestRadixNode:
    """Tests for RadixNode class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_node_initialization(self) -> None:
        """Test RadixNode initialization."""
        node = RadixNode(node_id=0, label=None)
        assert node.node_id == 0
        assert node.label is None
        assert node.visit_count == 0
        assert len(node.children) == 0
        assert node.parent is None

    def test_node_with_label(self) -> None:
        """Test RadixNode initialization with label."""
        node = RadixNode(node_id=1, label=512)
        assert node.node_id == 1
        assert node.label == 512

    # ============================================================================
    # Child Management Tests
    # ============================================================================

    def test_add_child(self) -> None:
        """Test adding a child to a node."""
        parent = RadixNode(node_id=0, label=None)
        child = RadixNode(node_id=1, label=512)

        parent.add_child(512, child)

        assert 512 in parent.children
        assert parent.children[512] is child
        assert child.parent is parent

    def test_get_child_exists(self) -> None:
        """Test getting an existing child."""
        parent = RadixNode(node_id=0, label=None)
        child = RadixNode(node_id=1, label=512)
        parent.add_child(512, child)

        retrieved = parent.get_child(512)
        assert retrieved is child

    def test_get_child_not_exists(self) -> None:
        """Test getting a non-existent child returns None."""
        parent = RadixNode(node_id=0, label=None)
        retrieved = parent.get_child(999)
        assert retrieved is None

    def test_is_leaf_true(self) -> None:
        """Test is_leaf returns True for node with no children."""
        node = RadixNode(node_id=0, label=None)
        assert node.is_leaf() is True

    def test_is_leaf_false(self) -> None:
        """Test is_leaf returns False for node with children."""
        parent = RadixNode(node_id=0, label=None)
        child = RadixNode(node_id=1, label=512)
        parent.add_child(512, child)
        assert parent.is_leaf() is False


class TestRadixTree:
    """Tests for RadixTree class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_tree_initialization(self) -> None:
        """Test RadixTree initialization."""
        tree = RadixTree()
        assert tree.root is not None
        assert tree.root.node_id == 0
        assert tree.root.is_leaf() is True

    # ============================================================================
    # Path Addition Tests
    # ============================================================================

    def test_add_single_path(self) -> None:
        """Test adding a single path to tree."""
        tree = RadixTree()
        leaf = tree.add_path([1, 2, 3])

        assert leaf is not None
        assert leaf.is_leaf() is True
        assert leaf.visit_count == 1

    def test_add_multiple_paths(self) -> None:
        """Test adding multiple paths to tree."""
        tree = RadixTree()
        leaf1 = tree.add_path([1, 2, 3])
        leaf2 = tree.add_path([1, 2, 4])

        assert leaf1 is not leaf2
        assert leaf1.is_leaf() is True
        assert leaf2.is_leaf() is True

    def test_add_shared_prefix(self) -> None:
        """Test that paths with shared prefixes reuse nodes."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])

        # Both paths start with [1, 2], so they should share nodes
        stats = tree.get_stats()
        # Should have created nodes for 1, 2, 3, 4 (4 non-root nodes minimum)
        assert stats.num_nodes >= 4

    def test_add_path_increments_visit_count(self) -> None:
        """Test that adding the same path increments visit count."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        leaf1 = tree.add_path([1, 2, 3])

        assert leaf1.visit_count == 2

    def test_add_empty_path(self) -> None:
        """Test adding an empty path."""
        tree = RadixTree()
        leaf = tree.add_path([])

        assert leaf is tree.root
        assert tree.root.visit_count == 1

    @pytest.mark.parametrize("path_length", [1, 5, 10, 50])
    def test_add_paths_various_lengths(self, path_length: int) -> None:
        """Test adding paths of various lengths."""
        tree = RadixTree()
        path = list(range(path_length))
        leaf = tree.add_path(path)

        assert leaf is not None
        assert leaf.visit_count == 1

    # ============================================================================
    # Node Retrieval Tests
    # ============================================================================

    def test_get_node_exists(self) -> None:
        """Test retrieving an existing node."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        root = tree.get_node(0)
        assert root is tree.root

    def test_get_node_not_exists(self) -> None:
        """Test retrieving a non-existent node returns None."""
        tree = RadixTree()
        node = tree.get_node(999)
        assert node is None

    def test_get_all_nodes(self) -> None:
        """Test getting all nodes in the tree."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])

        nodes = tree.get_all_nodes()
        assert len(nodes) >= 4  # At least root + 3 levels

    # ============================================================================
    # Statistics Tests
    # ============================================================================

    def test_get_stats_empty_tree(self) -> None:
        """Test statistics for empty tree."""
        tree = RadixTree()
        stats = tree.get_stats()

        assert isinstance(stats, RadixTreeStats)
        assert stats.num_nodes == 1  # Just root
        assert stats.num_leaves == 1  # Root is a leaf
        assert stats.total_visits == 0
        assert stats.max_depth == 0

    def test_get_stats_single_path(self) -> None:
        """Test statistics for tree with single path."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        stats = tree.get_stats()
        assert stats.num_nodes >= 4  # Root + 3 nodes
        assert stats.total_visits == 1
        assert stats.max_depth >= 3

    def test_get_stats_multiple_visits(self) -> None:
        """Test statistics count visits correctly."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 3])

        stats = tree.get_stats()
        assert stats.total_visits == 3

    # ============================================================================
    # Tree Structure Tests
    # ============================================================================

    def test_parent_child_relationships(self) -> None:
        """Test that parent-child relationships are consistent."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        nodes = tree.get_all_nodes()
        for node in nodes:
            if node.node_id != 0:  # Not root
                assert node.parent is not None

    def test_tree_is_connected(self) -> None:
        """Test that all nodes are reachable from root."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 4, 5])

        # All nodes should be reachable via DFS from root
        visited = set()

        def dfs(node: RadixNode) -> None:
            visited.add(node.node_id)
            for child in node.children.values():
                if child.node_id not in visited:
                    dfs(child)

        dfs(tree.root)
        all_nodes = {n.node_id for n in tree.get_all_nodes()}
        assert visited == all_nodes

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_large_tree(self) -> None:
        """Test building a large tree."""
        tree = RadixTree()

        for i in range(100):
            path = [i % 10, (i // 10) % 10, (i // 100) % 10]
            tree.add_path(path)

        stats = tree.get_stats()
        assert stats.num_nodes > 1
        assert stats.total_visits == 100

    def test_tree_with_single_branch(self) -> None:
        """Test tree with a single long branch."""
        tree = RadixTree()
        long_path = list(range(100))
        tree.add_path(long_path)

        stats = tree.get_stats()
        assert stats.max_depth >= 99
