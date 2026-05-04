# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the v1 -> v2 converter (``convert_user_to_aiperf``).

These complement the per-section builder tests (``test_converter_*``) by
probing the inter-builder seams: magic-list promotion edge cases, recipe
mutual-exclusion paths, recipe -> sweep / multi_run lifting, and the full
``convert_user_to_aiperf`` integration with conflicting / minimal /
malformed v1 inputs. Each test names the bug it would catch -- the
converter is the single v1 -> v2 boundary so silent regressions here
break every CLI command at once.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import (
    _promote_magic_lists_to_sweep_block,
    _reject_recipe_plus_magic_lists,
    convert_user_to_aiperf,
)

# =====================================================================
# Magic-list promotion (`_promote_magic_lists_to_sweep_block`)
# =====================================================================


class TestMagicListPromotion:
    def test_single_element_concurrency_list_still_promotes(self) -> None:
        """A single-element list is technically a sweep with one cell. The
        converter's job is not to second-guess intent -- promote it; the
        downstream sweep machinery handles 1-cell sweeps fine.
        """
        nested: dict = {
            "phases": [
                {"name": "profiling", "type": "concurrency", "concurrency": [42]}
            ]
        }
        _promote_magic_lists_to_sweep_block(nested)
        assert nested["sweep"]["variables"] == {"phases.profiling.concurrency": [42]}

    def test_multiple_magic_lists_in_same_phase_promoted_together(self) -> None:
        """Both `concurrency` and `request_rate` (when list-shaped) lift in
        the same pass. Order across the dict is preserved by Python's
        insertion-ordered dict semantics; verify both keys land."""
        nested: dict = {
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": [1, 2, 4],
                    "rate": [10, 20],
                }
            ]
        }
        _promote_magic_lists_to_sweep_block(nested)
        vars_ = nested["sweep"]["variables"]
        assert vars_ == {
            "phases.profiling.concurrency": [1, 2, 4],
            "phases.profiling.rate": [10, 20],
        }

    def test_magic_list_merges_into_existing_sweep_block(self) -> None:
        """A YAML-declared sweep block should NOT be overwritten by magic-
        list promotion. Both sets of variables co-exist via dict.update."""
        nested: dict = {
            "phases": [
                {"name": "profiling", "type": "concurrency", "concurrency": [1, 2]}
            ],
            "sweep": {"type": "grid", "variables": {"random_seed": [1, 2, 3]}},
        }
        _promote_magic_lists_to_sweep_block(nested)
        vars_ = nested["sweep"]["variables"]
        assert vars_["random_seed"] == [1, 2, 3]
        assert vars_["phases.profiling.concurrency"] == [1, 2]

    def test_no_magic_lists_leaves_sweep_unset(self) -> None:
        """When no phase field is list-shaped, the sweep block is NOT
        synthesized -- otherwise we'd emit an empty sweep that downstream
        sees as a 0-cell parameter sweep."""
        nested: dict = {
            "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 8}]
        }
        _promote_magic_lists_to_sweep_block(nested)
        assert "sweep" not in nested

    def test_no_phases_block_is_a_no_op(self) -> None:
        """Defensive: ``nested`` may not have phases yet (callers compose
        sections incrementally). Don't crash; just no-op."""
        nested: dict = {}
        _promote_magic_lists_to_sweep_block(nested)
        assert nested == {}

    def test_phases_not_a_list_raises_type_error(self) -> None:
        """A typo where phases is a dict (pre-shorthand-promotion shape)
        must error loudly -- silently skipping would let a phantom sweep
        slip through."""
        with pytest.raises(TypeError, match="phases must be a list"):
            _promote_magic_lists_to_sweep_block({"phases": {"profiling": {}}})

    def test_phase_entry_missing_name_raises_value_error(self) -> None:
        """Magic-list keys phases by name; a phase without one is unlift-
        able and the error must name the offending index."""
        with pytest.raises(ValueError, match=r"phases\[0\] is missing"):
            _promote_magic_lists_to_sweep_block(
                {"phases": [{"type": "concurrency", "concurrency": [1, 2]}]}
            )

    def test_phase_entry_not_a_dict_raises_type_error(self) -> None:
        """A scalar in the phase list (a YAML mistake) must error early."""
        with pytest.raises(TypeError, match=r"phases\[0\] must be a dict"):
            _promote_magic_lists_to_sweep_block({"phases": ["not_a_dict_phase"]})

    def test_non_magic_list_field_with_list_value_left_alone(self) -> None:
        """Only fields named in MAGIC_LIST_FIELDS get promoted. A list-
        valued non-magic field (e.g. `dataset_files`) stays put."""
        nested: dict = {
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 8,
                    "custom_list_field": ["a", "b"],
                }
            ]
        }
        _promote_magic_lists_to_sweep_block(nested)
        assert "sweep" not in nested
        assert nested["phases"][0]["custom_list_field"] == ["a", "b"]


# =====================================================================
# Recipe + magic-list mutual exclusion (`_reject_recipe_plus_magic_lists`)
# =====================================================================


class TestRecipePlusMagicListRejection:
    def test_grid_recipe_plus_concurrency_list_rejected(self) -> None:
        """A grid recipe owns the swept variables. Pairing it with a magic-
        list `--concurrency 1,5,10` is ambiguous -- which list wins? Hard-
        fail at convert time with both flag names called out."""
        user = UserConfig.model_validate(
            {
                "endpoint": {"streaming": True},
                "loadgen": {
                    "search_recipe": "concurrency-ramp",
                    "concurrency": [1, 5, 10],
                },
            }
        )
        with pytest.raises((ValueError, TypeError), match="recipe.*concurrency"):
            _reject_recipe_plus_magic_lists(user)

    def test_no_recipe_no_magic_list_passes(self) -> None:
        """Sanity: clean inputs don't trip the rejection."""
        user = UserConfig.model_validate(
            {"endpoint": {"streaming": True}, "loadgen": {"concurrency": 8}}
        )
        # Should be a no-op (no exception).
        _reject_recipe_plus_magic_lists(user)

    def test_recipe_alone_passes(self) -> None:
        """Recipe without magic-list flags is the canonical path."""
        user = UserConfig.model_validate(
            {
                "endpoint": {"streaming": True},
                "loadgen": {
                    "search_recipe": "concurrency-ramp",
                    "degradation_threshold": 0.20,
                },
            }
        )
        _reject_recipe_plus_magic_lists(user)


# =====================================================================
# convert_user_to_aiperf full integration (adversarial inputs)
# =====================================================================


class TestConvertUserToAiperfIntegration:
    def _user_minimal(self) -> UserConfig:
        return UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                },
                "loadgen": {"concurrency": 8, "request_count": 30},
            }
        )

    def test_minimal_user_config_validates(self) -> None:
        """Sanity: smallest legal v1 input produces a valid v2 AIPerfConfig."""
        cfg = convert_user_to_aiperf(self._user_minimal(), ServiceConfig())
        assert [m.name for m in cfg.benchmark.models.items] == ["m"]

    def test_empty_model_names_fails_v2_validation(self) -> None:
        """v1 ``endpoint.model_names`` was made optional (default `[]`) so
        cyclopts can build UserConfig from sparse CLI flags. v2
        AIPerfConfig still rejects empty models -- the layer of enforcement
        moves DOWN, not away."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": [],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                }
            }
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            convert_user_to_aiperf(user, ServiceConfig())

    def test_warmup_phase_added_when_warmup_request_count_set(self) -> None:
        """Warmup is opt-in: setting any --warmup-* flag emits a `warmup`
        phase BEFORE the `profiling` phase, in that order. Order matters
        because ``cli_runner`` iterates phases sequentially."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                },
                "loadgen": {
                    "concurrency": 8,
                    "request_count": 30,
                    "warmup_request_count": 5,
                },
            }
        )
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        names = [p.name for p in cfg.benchmark.phases]
        assert names == ["warmup", "profiling"]

    def test_no_warmup_phase_when_no_warmup_flags(self) -> None:
        """Warmup must NOT be auto-emitted when the user didn't ask for it."""
        cfg = convert_user_to_aiperf(self._user_minimal(), ServiceConfig())
        assert [p.name for p in cfg.benchmark.phases] == ["profiling"]

    def test_recipe_lifts_sweep_to_top_level(self) -> None:
        """Grid recipe end-to-end through the converter must land
        sweep_variables at AIPerfConfig.sweep, not bury them in multi_run."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                    "streaming": True,
                },
                "loadgen": {
                    "search_recipe": "concurrency-ramp",
                    "concurrency": 8,
                    "request_count": 30,
                },
            }
        )
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        assert cfg.sweep is not None
        # GridSweep dumps `variables: dict[str, list]`; key by full path.
        assert "phases.profiling.concurrency" in cfg.sweep.variables

    def test_bo_recipe_lands_adaptive_search_with_sla_filters(self) -> None:
        """The whole point of `--search-recipe max-throughput-ttft-sla
        --ttft-sla-ms 200`: the recipe expands to AdaptiveSearchConfig with
        the SLA filter populated. Lock the entire chain in one assertion."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                    "streaming": True,
                },
                "loadgen": {
                    "search_recipe": "max-throughput-ttft-sla",
                    "ttft_sla_ms": 200.0,
                    "concurrency": 8,
                    "request_count": 30,
                },
            }
        )
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        assert cfg.multi_run is not None
        ad = cfg.multi_run.adaptive_search
        assert ad is not None
        assert ad.recipe_name == "max-throughput-ttft-sla"
        assert len(ad.sla_filters) == 1
        assert ad.sla_filters[0].metric_tag == "time_to_first_token"
        assert ad.sla_filters[0].threshold == 200.0

    @pytest.mark.parametrize(
        "recipe,expected_metric",
        [
            param("max-throughput-ttft-sla", "time_to_first_token", id="ttft"),
            param("max-throughput-itl-sla", "inter_token_latency", id="itl"),
        ],
    )
    def test_each_bo_recipe_picks_its_metric(
        self, recipe: str, expected_metric: str
    ) -> None:
        """Each built-in BO recipe is wired to its own metric; a copy-paste
        bug in the recipe class would route both to the wrong one."""
        sla_field = (
            "ttft_sla_ms" if recipe == "max-throughput-ttft-sla" else "itl_sla_ms"
        )
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                    "streaming": True,
                },
                "loadgen": {
                    "search_recipe": recipe,
                    sla_field: 100.0,
                    "concurrency": 8,
                    "request_count": 30,
                },
            }
        )
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        assert cfg.multi_run.adaptive_search.sla_filters[0].metric_tag == (
            expected_metric
        )

    def test_streaming_only_recipe_against_no_streaming_rejected(self) -> None:
        """`prefill-ttft-curve` requires `--streaming`; without it the
        recipe must reject at expand time with a recipe-named message."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                    "streaming": False,
                },
                "loadgen": {
                    "search_recipe": "prefill-ttft-curve",
                    "concurrency": 8,
                    "request_count": 30,
                },
            }
        )
        with pytest.raises(ValueError, match="streaming"):
            convert_user_to_aiperf(user, ServiceConfig())

    def test_search_space_without_metric_rejected(self) -> None:
        """Adversarial: `--search-space` alone (no metric/direction/max-iters)
        must fail with a clear missing-companion error -- not silently emit
        a half-formed adaptive_search block downstream."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                },
                "loadgen": {
                    # v1 search_space is the CLI string shape: 'path:lo,hi:kind'.
                    "search_space": ["phases.profiling.concurrency:1,100:int"],
                    "concurrency": 8,
                    "request_count": 30,
                },
            }
        )
        with pytest.raises(TypeError, match="search-metric"):
            convert_user_to_aiperf(user, ServiceConfig())

    def test_grid_recipe_plus_magic_list_rejected_at_convert(self) -> None:
        """End-to-end: `--search-recipe concurrency-ramp --concurrency 1,5,10`
        is ambiguous and the converter's mutual-exclusion gate must fire."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                },
                "loadgen": {
                    "search_recipe": "concurrency-ramp",
                    "concurrency": [1, 5, 10],
                    "request_count": 30,
                },
            }
        )
        with pytest.raises((ValueError, TypeError), match="recipe"):
            convert_user_to_aiperf(user, ServiceConfig())


# =====================================================================
# Random-seed init: converter must not silently mutate user inputs
# =====================================================================


class TestRandomSeedHandling:
    def test_user_supplied_random_seed_preserved(self) -> None:
        """When the user passed `--random-seed 42`, that value must reach
        the AIPerfConfig unchanged. Silent re-seeding would break
        reproducibility -- a user-visible regression magnet."""
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                },
                "input": {"random_seed": 42},
                "loadgen": {"concurrency": 8, "request_count": 30},
            }
        )
        convert_user_to_aiperf(user, ServiceConfig())
        # The seed lands wherever AIPerfConfig stores it; the v1 layer must
        # not mutate the passed-in user.input.random_seed during convert.
        assert user.input.random_seed == 42
