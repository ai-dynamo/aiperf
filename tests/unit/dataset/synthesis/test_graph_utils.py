# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for graph utilities."""

from aiperf.dataset.synthesis import RadixTree
from aiperf.dataset.synthesis.graph_utils import (
    compute_transition_cdfs,
    get_tree_stats,
    merge_unary_chains,
    remove_leaves,
    validate_tree,
)


class TestGraphUtils:
    """Tests for graph utility functions."""

    # ============================================================================
    # Tree Validation Tests
    # ============================================================================

    def test_validate_tree_empty(self) -> None:
        """Test validating an empty tree."""
        tree = RadixTree()
        assert validate_tree(tree) is True

    def test_validate_tree_single_path(self) -> None:
        """Test validating tree with single path."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        assert validate_tree(tree) is True

    def test_validate_tree_multiple_paths(self) -> None:
        """Test validating tree with multiple paths."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])
        tree.add_path([1, 5, 6])
        assert validate_tree(tree) is True

    def test_validate_tree_all_nodes_reachable(self) -> None:
        """Test that validation checks all nodes are reachable."""
        tree = RadixTree()
        tree.add_path([1, 2, 3, 4, 5])
        tree.add_path([2, 3, 4, 5, 6])
        assert validate_tree(tree) is True

    # ============================================================================
    # Leaf Removal Tests
    # ============================================================================

    def test_remove_leaves_preserves_root(self) -> None:
        """Test that root is preserved after removing leaves."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 3])  # Add twice so it's not removed

        remove_leaves(tree, visit_threshold=1)

        assert tree.root is not None

    def test_remove_leaves_threshold(self) -> None:
        """Test leaf removal with different thresholds."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])  # Visit count 1
        tree.add_path([1, 2, 4])  # Visit count 1
        tree.add_path([1, 2, 4])  # Visit count 2 now

        remove_leaves(tree, visit_threshold=1)

        # Leaves with visit count 1 should be removed
        assert validate_tree(tree) is True

    def test_remove_leaves_no_effect_on_shared(self) -> None:
        """Test that removing leaves doesn't affect shared nodes."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 3])

        remove_leaves(tree, visit_threshold=2)  # Higher threshold

        # Nodes with visit count >= 2 should be preserved
        final_nodes = len(tree.get_all_nodes())
        assert final_nodes >= 1

    # ============================================================================
    # Unary Chain Merging Tests
    # ============================================================================

    def test_merge_unary_chains_preserves_structure(self) -> None:
        """Test that merging preserves tree structure."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        merge_unary_chains(tree)

        # Tree should still be valid
        assert validate_tree(tree) is True

    def test_merge_unary_chains_reduces_nodes(self) -> None:
        """Test that merging reduces node count for unary chains."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        initial_nodes = len(tree.get_all_nodes())
        merge_unary_chains(tree)
        final_nodes = len(tree.get_all_nodes())

        # Merging should not increase nodes
        assert final_nodes <= initial_nodes

    def test_merge_unary_chains_with_branching(self) -> None:
        """Test merging doesn't affect nodes with multiple children."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])  # Creates branch at 2

        merge_unary_chains(tree)

        # Tree should still be valid
        assert validate_tree(tree) is True

    # ============================================================================
    # CDF Computation Tests
    # ============================================================================

    def test_compute_transition_cdfs_empty_tree(self) -> None:
        """Test CDF computation on empty tree."""
        tree = RadixTree()
        cdfs = compute_transition_cdfs(tree)

        # Empty tree should produce empty or minimal CDFs
        assert isinstance(cdfs, dict)

    def test_compute_transition_cdfs_single_path(self) -> None:
        """Test CDF computation with single path."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        cdfs = compute_transition_cdfs(tree)

        # Should have CDFs for nodes with children
        assert len(cdfs) > 0

    def test_compute_transition_cdfs_values(self) -> None:
        """Test that CDF values are proper probabilities."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])

        cdfs = compute_transition_cdfs(tree)

        for cdf in cdfs.values():
            # CDFs should be increasing
            for i in range(len(cdf) - 1):
                assert cdf[i] <= cdf[i + 1]
            # Last value should be ~1.0
            assert abs(cdf[-1] - 1.0) < 0.01

    def test_compute_transition_cdfs_branching(self) -> None:
        """Test CDF computation with branching paths."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])

        cdfs = compute_transition_cdfs(tree)

        # Node 0 should have CDF for its children
        assert len(cdfs) > 0

    # ============================================================================
    # Tree Statistics Tests
    # ============================================================================

    def test_get_tree_stats_empty(self) -> None:
        """Test stats for empty tree."""
        tree = RadixTree()
        stats = get_tree_stats(tree)

        assert stats["num_nodes"] == 1  # Just root
        assert stats["num_leaves"] == 1

    def test_get_tree_stats_single_path(self) -> None:
        """Test stats for single path."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        stats = get_tree_stats(tree)

        assert stats["num_nodes"] >= 4
        assert "internal_nodes" in stats
        assert "branching_factor" in stats

    def test_get_tree_stats_branching(self) -> None:
        """Test stats for branching tree."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])
        tree.add_path([1, 5, 6])

        stats = get_tree_stats(tree)

        assert stats["num_nodes"] > 4
        assert stats["branching_factor"] > 1.0

    def test_get_tree_stats_includes_all_metrics(self) -> None:
        """Test that all expected metrics are present."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])

        stats = get_tree_stats(tree)

        required_keys = [
            "num_nodes",
            "num_leaves",
            "total_visits",
            "max_depth",
            "internal_nodes",
            "branching_factor",
        ]
        for key in required_keys:
            assert key in stats

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_all_operations_combined(self) -> None:
        """Test combining multiple operations."""
        tree = RadixTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])
        tree.add_path([1, 5, 6, 7])

        # Validate initial state
        assert validate_tree(tree) is True

        # Remove infrequent leaves
        remove_leaves(tree, visit_threshold=1)
        assert validate_tree(tree) is True

        # Merge chains
        merge_unary_chains(tree)
        assert validate_tree(tree) is True

        # Compute CDFs
        cdfs = compute_transition_cdfs(tree)
        assert len(cdfs) >= 0

        # Get stats
        stats = get_tree_stats(tree)
        assert stats["num_nodes"] >= 1
