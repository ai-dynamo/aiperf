# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for sweep expansion that are NOT covered by test_sweep.py.

Focuses on:
- Grid sweep: single-element lists, falsy/None values, deep nesting, large products
- Scenario sweep: empty dicts, name fallback, deep-merge sibling preservation
- Magic list detection: mixed types, empties, nested fields, non-magic names
- _set_nested_value: scalar overwrite, repeated keys, single-key paths
- _deep_merge: list replacement, None overrides, new key addition, deep nesting
"""

import pytest

from aiperf.config.sweep import (
    MAGIC_LIST_FIELDS,
    _deep_merge,
    _set_nested_value,
    detect_sweep_fields,
    expand_sweep,
)

# ============================================================
# Grid Sweep Edge Cases
# ============================================================


class TestGridSweepEdgeCases:
    """Boundary conditions for grid (Cartesian product) expansion."""

    def test_single_element_list_produces_one_variation(self) -> None:
        data = {"sweep": {"type": "grid", "variables": {"x": [5]}}}
        result = expand_sweep(data)

        assert len(result) == 1
        cfg, var = result[0]
        assert cfg["x"] == 5
        assert var.index == 0
        assert var.values == {"x": 5}

    def test_falsy_values_all_preserved(self) -> None:
        """0, False, and 0.0 are all valid sweep values."""
        data = {"sweep": {"type": "grid", "variables": {"x": [0, False, 0.0]}}}
        result = expand_sweep(data)

        assert len(result) == 3
        values = [r[0]["x"] for r in result]
        assert values == [0, False, 0.0]

    def test_none_value_preserved_in_list(self) -> None:
        data = {"sweep": {"type": "grid", "variables": {"x": [1, None, 3]}}}
        result = expand_sweep(data)

        assert len(result) == 3
        values = [r[0]["x"] for r in result]
        assert values == [1, None, 3]

    def test_deeply_nested_path(self) -> None:
        data = {"sweep": {"type": "grid", "variables": {"a.b.c.d.e": [1, 2]}}}
        result = expand_sweep(data)

        assert len(result) == 2
        assert result[0][0]["a"]["b"]["c"]["d"]["e"] == 1
        assert result[1][0]["a"]["b"]["c"]["d"]["e"] == 2

    def test_multiple_variables_same_parent(self) -> None:
        """Two variables under the same parent produce correct Cartesian product."""
        data = {
            "parent": {"x": 0, "y": 0},
            "sweep": {
                "type": "grid",
                "variables": {"parent.x": [1, 2], "parent.y": [3, 4]},
            },
        }
        result = expand_sweep(data)

        assert len(result) == 4
        pairs = {(r[0]["parent"]["x"], r[0]["parent"]["y"]) for r in result}
        assert pairs == {(1, 3), (1, 4), (2, 3), (2, 4)}

    def test_large_cartesian_product(self) -> None:
        data = {
            "sweep": {
                "type": "grid",
                "variables": {
                    "a": list(range(5)),
                    "b": list(range(5)),
                    "c": list(range(5)),
                },
            },
        }
        result = expand_sweep(data)

        assert len(result) == 125
        # Indices are sequential
        indices = [r[1].index for r in result]
        assert indices == list(range(125))

    def test_label_format_multi_variable(self) -> None:
        data = {
            "sweep": {
                "type": "grid",
                "variables": {"alpha": [1], "beta": [2]},
            },
        }
        result = expand_sweep(data)

        assert len(result) == 1
        assert result[0][1].label == "alpha=1, beta=2"

    def test_sweep_key_stripped_from_all_variations(self) -> None:
        data = {
            "keep": "this",
            "sweep": {"type": "grid", "variables": {"x": [1, 2, 3]}},
        }
        result = expand_sweep(data)

        for cfg, _ in result:
            assert "sweep" not in cfg
            assert cfg["keep"] == "this"


# ============================================================
# Scenario Sweep Edge Cases
# ============================================================


class TestScenarioSweepEdgeCases:
    """Boundary conditions for scenario (deep-merge) expansion."""

    def test_empty_scenario_dict_leaves_base_unchanged(self) -> None:
        data = {
            "base_key": "original",
            "sweep": {"type": "scenarios", "runs": [{}]},
        }
        result = expand_sweep(data)

        assert len(result) == 1
        cfg, var = result[0]
        assert cfg["base_key"] == "original"
        assert var.values == {}

    def test_scenario_with_only_name_no_config_changes(self) -> None:
        data = {
            "base_key": "original",
            "sweep": {"type": "scenarios", "runs": [{"name": "test-only"}]},
        }
        result = expand_sweep(data)

        assert len(result) == 1
        cfg, var = result[0]
        assert var.label == "test-only"
        assert cfg["base_key"] == "original"
        # "name" is stripped from scenario_data, so values should be empty
        assert var.values == {}

    def test_scenario_label_fallback_to_index(self) -> None:
        data = {
            "sweep": {
                "type": "scenarios",
                "runs": [{"x": 1}, {"x": 2}, {"x": 3}],
            },
        }
        result = expand_sweep(data)

        labels = [r[1].label for r in result]
        assert labels == ["scenario_0", "scenario_1", "scenario_2"]

    def test_scenario_deep_merge_preserves_sibling_keys(self) -> None:
        data = {
            "phases": {"concurrency": 8, "requests": 100, "rate": 5.0},
            "sweep": {
                "type": "scenarios",
                "runs": [{"phases": {"concurrency": 32}}],
            },
        }
        result = expand_sweep(data)

        cfg = result[0][0]
        assert cfg["phases"]["concurrency"] == 32
        assert cfg["phases"]["requests"] == 100
        assert cfg["phases"]["rate"] == 5.0

    def test_scenario_adds_new_nested_keys(self) -> None:
        """Scenario can introduce fields that don't exist in base."""
        data = {
            "phases": {"concurrency": 8},
            "sweep": {
                "type": "scenarios",
                "runs": [{"phases": {"new_field": "added"}, "extra": {"deep": True}}],
            },
        }
        result = expand_sweep(data)

        cfg = result[0][0]
        assert cfg["phases"]["concurrency"] == 8
        assert cfg["phases"]["new_field"] == "added"
        assert cfg["extra"]["deep"] is True

    def test_scenario_variations_are_independent(self) -> None:
        """Mutations in one variation must not leak into another."""
        data = {
            "shared": {"val": "original"},
            "sweep": {
                "type": "scenarios",
                "runs": [
                    {"shared": {"val": "first"}},
                    {"shared": {"val": "second"}},
                ],
            },
        }
        result = expand_sweep(data)

        assert result[0][0]["shared"]["val"] == "first"
        assert result[1][0]["shared"]["val"] == "second"


# ============================================================
# Magic List Detection Edge Cases
# ============================================================


class TestMagicListEdgeCases:
    """Boundary conditions for detect_sweep_fields and magic list expansion."""

    def test_mixed_int_float_list_detected(self) -> None:
        data = {"phases": {"concurrency": [1, 2.5, 3]}}
        fields = detect_sweep_fields(data)

        assert "phases.concurrency" in fields
        assert fields["phases.concurrency"] == [1, 2.5, 3]

    def test_float_only_list_detected(self) -> None:
        data = {"phases": {"rate": [1.5, 2.7]}}
        fields = detect_sweep_fields(data)

        assert "phases.rate" in fields

    def test_zero_in_list_detected(self) -> None:
        data = {"phases": {"concurrency": [0, 1, 2]}}
        fields = detect_sweep_fields(data)

        assert "phases.concurrency" in fields
        assert fields["phases.concurrency"] == [0, 1, 2]

    def test_empty_list_detected_but_produces_base_variation(self) -> None:
        """Empty list passes detection but itertools.product yields nothing."""
        data = {
            "models": ["m"],
            "phases": {
                "default": {"type": "concurrency", "concurrency": [], "requests": 10}
            },
        }
        fields = detect_sweep_fields(data)
        assert "phases.default.concurrency" in fields

        result = expand_sweep(data)
        assert len(result) == 1
        assert result[0][1].label == "base"

    def test_string_list_not_detected(self) -> None:
        data = {"phases": {"concurrency": ["8", "16"]}}
        fields = detect_sweep_fields(data)

        assert len(fields) == 0

    def test_mixed_string_and_int_not_detected(self) -> None:
        """all() check rejects lists with any non-numeric element."""
        data = {"phases": {"concurrency": [1, "2", 3]}}
        fields = detect_sweep_fields(data)

        assert len(fields) == 0

    def test_nested_magic_field_detected(self) -> None:
        data = {"phases": {"warmup": {"concurrency": [1, 2]}}}
        fields = detect_sweep_fields(data)

        assert "phases.warmup.concurrency" in fields

    def test_non_magic_field_name_ignored(self) -> None:
        data = {"phases": {"custom_param": [1, 2, 3]}}
        fields = detect_sweep_fields(data)

        assert len(fields) == 0

    @pytest.mark.parametrize(
        "field_name",
        sorted(MAGIC_LIST_FIELDS),
    )  # fmt: skip
    def test_all_magic_field_names_recognized(self, field_name: str) -> None:
        data = {"section": {field_name: [1, 2]}}
        fields = detect_sweep_fields(data)

        assert f"section.{field_name}" in fields

    def test_bool_in_list_passes_numeric_check(self) -> None:
        """bool is a subclass of int, so True/False pass the isinstance check."""
        data = {"phases": {"concurrency": [True, False]}}
        fields = detect_sweep_fields(data)

        assert "phases.concurrency" in fields

    def test_none_in_list_not_detected(self) -> None:
        """None is not numeric, so the list fails the all() check."""
        data = {"phases": {"concurrency": [1, None, 3]}}
        fields = detect_sweep_fields(data)

        assert len(fields) == 0

    def test_magic_list_expansion_produces_correct_variations(self) -> None:
        """End-to-end: magic list detected and expanded into separate variations."""
        data = {
            "models": ["m"],
            "phases": {"default": {"type": "concurrency", "concurrency": [4, 8, 16]}},
        }
        result = expand_sweep(data)

        assert len(result) == 3
        values = [r[0]["phases"]["default"]["concurrency"] for r in result]
        assert values == [4, 8, 16]

    def test_multiple_magic_lists_produce_cartesian_product(self) -> None:
        data = {
            "phases": {
                "default": {
                    "concurrency": [1, 2],
                    "rate": [10.0, 20.0],
                }
            }
        }
        result = expand_sweep(data)

        assert len(result) == 4
        pairs = {
            (
                r[0]["phases"]["default"]["concurrency"],
                r[0]["phases"]["default"]["rate"],
            )
            for r in result
        }
        assert pairs == {(1, 10.0), (1, 20.0), (2, 10.0), (2, 20.0)}


# ============================================================
# _set_nested_value Edge Cases
# ============================================================


class TestSetNestedValueEdgeCases:
    """Boundary conditions for dot-notation path setting."""

    def test_single_key_path(self) -> None:
        data: dict = {}
        _set_nested_value(data, "x", 42)

        assert data["x"] == 42

    def test_repeated_keys_in_path(self) -> None:
        data: dict = {}
        _set_nested_value(data, "a.a.a", "deep")

        assert data == {"a": {"a": {"a": "deep"}}}

    def test_overwrite_scalar_at_intermediate_path_raises(self) -> None:
        """Cannot traverse through a scalar to create children."""
        data = {"a": 5}
        with pytest.raises(TypeError):
            _set_nested_value(data, "a.b", 10)

    def test_creates_all_intermediate_dicts(self) -> None:
        data: dict = {}
        _set_nested_value(data, "a.b.c.d.e", "leaf")

        assert data["a"]["b"]["c"]["d"]["e"] == "leaf"

    def test_preserves_existing_siblings(self) -> None:
        data = {"parent": {"existing": "keep", "target": "old"}}
        _set_nested_value(data, "parent.target", "new")

        assert data["parent"]["target"] == "new"
        assert data["parent"]["existing"] == "keep"

    def test_overwrites_existing_value(self) -> None:
        data = {"a": {"b": "old"}}
        _set_nested_value(data, "a.b", "new")

        assert data["a"]["b"] == "new"

    def test_sets_none_value(self) -> None:
        data: dict = {}
        _set_nested_value(data, "a.b", None)

        assert data["a"]["b"] is None

    def test_sets_complex_value(self) -> None:
        data: dict = {}
        _set_nested_value(data, "a.b", {"nested": [1, 2, 3]})

        assert data["a"]["b"] == {"nested": [1, 2, 3]}


# ============================================================
# _deep_merge Edge Cases
# ============================================================


class TestDeepMergeEdgeCases:
    """Boundary conditions for recursive dictionary merging."""

    def test_list_replaced_not_concatenated(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": [4]}
        _deep_merge(base, override)

        assert base["items"] == [4]

    def test_none_override_replaces_value(self) -> None:
        base = {"x": 5}
        override = {"x": None}
        _deep_merge(base, override)

        assert base["x"] is None

    def test_new_keys_added(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        _deep_merge(base, override)

        assert base == {"a": 1, "b": 2}

    def test_nested_merge_three_levels(self) -> None:
        base = {"l1": {"l2": {"l3": "old", "sibling": "keep"}}}
        override = {"l1": {"l2": {"l3": "new"}}}
        _deep_merge(base, override)

        assert base["l1"]["l2"]["l3"] == "new"
        assert base["l1"]["l2"]["sibling"] == "keep"

    def test_empty_override_is_noop(self) -> None:
        base = {"a": 1, "b": {"c": 2}}
        original = {"a": 1, "b": {"c": 2}}
        _deep_merge(base, {})

        assert base == original

    def test_empty_base_accepts_all_overrides(self) -> None:
        base: dict = {}
        override = {"a": {"b": {"c": 1}}}
        _deep_merge(base, override)

        assert base == {"a": {"b": {"c": 1}}}

    def test_dict_override_replaces_scalar(self) -> None:
        """When base has scalar and override has dict, dict wins."""
        base = {"x": 5}
        override = {"x": {"nested": True}}
        _deep_merge(base, override)

        assert base["x"] == {"nested": True}

    def test_scalar_override_replaces_dict(self) -> None:
        """When base has dict and override has scalar, scalar wins."""
        base = {"x": {"nested": True}}
        override = {"x": 42}
        _deep_merge(base, override)

        assert base["x"] == 42

    def test_multiple_branches_merged_independently(self) -> None:
        base = {
            "branch_a": {"val": 1, "keep": True},
            "branch_b": {"val": 2, "keep": True},
        }
        override = {
            "branch_a": {"val": 10},
            "branch_b": {"val": 20},
        }
        _deep_merge(base, override)

        assert base["branch_a"] == {"val": 10, "keep": True}
        assert base["branch_b"] == {"val": 20, "keep": True}
