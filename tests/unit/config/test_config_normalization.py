# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for before-validators that normalize shorthand config input.

Focuses on:
- models: string / list[str] / singular "model" key -> ModelsAdvanced
- datasets: singular "dataset" key -> wrapped under "default", default type
- load: single phase dict (has "type") -> wrapped under "default", phase name injection
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.config import BenchmarkConfig
from aiperf.config.dataset import FileDataset, SyntheticDataset
from aiperf.config.models import ModelsAdvanced

_ENDPOINT = {"urls": ["http://localhost:8000/v1/chat/completions"]}
_SYNTHETIC_DATASET = {
    "type": "synthetic",
    "entries": 100,
    "prompts": {"isl": 128, "osl": 64},
}
_DEFAULT_NAMED_DATASET = {"name": "profiling", **_SYNTHETIC_DATASET}
_CONCURRENCY_PHASE = {"type": "concurrency", "requests": 10, "concurrency": 1}


def _minimal(**overrides: object) -> dict:
    """Minimal valid BenchmarkConfig dict with overrides."""
    base: dict = {
        "models": ["m"],
        "endpoint": _ENDPOINT,
        "datasets": [_DEFAULT_NAMED_DATASET],
        "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
    }
    base.update(overrides)
    return base


# ============================================================
# Model Normalization
# ============================================================


class TestModelNormalization:
    """Verify normalize_before_validation handles models shorthand forms."""

    def test_string_model_normalized_to_models_advanced(self) -> None:
        cfg = BenchmarkConfig.model_validate(_minimal(models="gpt-4"))

        assert isinstance(cfg.models, ModelsAdvanced)
        assert len(cfg.models.items) == 1
        assert cfg.models.items[0].name == "gpt-4"

    def test_list_of_strings_normalized(self) -> None:
        cfg = BenchmarkConfig.model_validate(_minimal(models=["gpt-4", "gpt-3.5"]))

        assert isinstance(cfg.models, ModelsAdvanced)
        assert len(cfg.models.items) == 2
        names = [item.name for item in cfg.models.items]
        assert names == ["gpt-4", "gpt-3.5"]

    def test_singular_model_key_accepted(self) -> None:
        data = _minimal()
        del data["models"]
        data["model"] = "llama-3"

        cfg = BenchmarkConfig.model_validate(data)

        assert len(cfg.models.items) == 1
        assert cfg.models.items[0].name == "llama-3"

    def test_already_structured_models_passthrough(self) -> None:
        structured = {
            "strategy": "round_robin",
            "items": [
                {"name": "llama-3", "weight": None},
                {"name": "mistral-7b", "weight": None},
            ],
        }
        cfg = BenchmarkConfig.model_validate(_minimal(models=structured))

        assert len(cfg.models.items) == 2
        assert cfg.models.items[0].name == "llama-3"
        assert cfg.models.items[1].name == "mistral-7b"

    def test_singular_model_with_plural_models_rejected(self) -> None:
        """`model` and `models` together are mutually exclusive — like `dataset`/`datasets`."""
        data = _minimal(models=["keep-me"])
        data["model"] = "ignore-me"

        with pytest.raises(
            ValidationError, match="'model' cannot be used with 'models'"
        ):
            BenchmarkConfig.model_validate(data)


# ============================================================
# Dataset Normalization
# ============================================================


class TestDatasetNormalization:
    """Verify parse_datasets and normalize_before_validation for datasets."""

    def test_singular_dataset_key_wrapped(self) -> None:
        data = _minimal()
        del data["datasets"]
        data["dataset"] = _SYNTHETIC_DATASET

        cfg = BenchmarkConfig.model_validate(data)

        assert "default" in [d.name for d in cfg.datasets]
        assert isinstance(cfg.datasets[0], SyntheticDataset)

    def test_file_dataset_with_osl_roundtrips(self) -> None:
        file_dict = {
            "type": "file",
            "path": "/tmp/data.jsonl",
            "osl": {"mean": 128, "stddev": 20},
        }
        cfg = BenchmarkConfig.model_validate(
            _minimal(datasets=[{"name": "mixed", **file_dict}])
        )

        ds = cfg.datasets[0]
        assert isinstance(ds, FileDataset)
        assert ds.osl is not None
        assert ds.osl.mean == 128
        assert ds.osl.stddev == 20

    def test_legacy_composed_shape_rejected(self) -> None:
        """source+augment shape was the v1 ComposedDataset; it's gone in v2."""
        legacy_dict = {
            "source": {
                "type": "file",
                "path": "/tmp/data.jsonl",
            },
            "augment": {
                "osl": {"mean": 128, "stddev": 20},
            },
        }
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            BenchmarkConfig.model_validate(
                _minimal(datasets=[{"name": "mixed", **legacy_dict}])
            )

    def test_default_type_synthetic(self) -> None:
        no_type = {"entries": 50, "prompts": {"isl": 64}}
        cfg = BenchmarkConfig.model_validate(
            _minimal(datasets=[{"name": "gen", **no_type}])
        )

        assert isinstance(cfg.datasets[0], SyntheticDataset)

    def test_explicit_type_preserved(self) -> None:
        cfg = BenchmarkConfig.model_validate(
            _minimal(
                datasets=[
                    {
                        "name": "trace",
                        "type": "file",
                        "path": "/tmp/trace.jsonl",
                        "format": "mooncake_trace",
                    }
                ]
            )
        )

        assert cfg.datasets[0].type == "file"


# ============================================================
# Load Normalization
# ============================================================


class TestLoadNormalization:
    """Verify normalize_before_validation and parse_load for load section."""

    def test_single_phase_wrapped_with_default_key(self) -> None:
        flat_load = {"type": "concurrency", "duration": 60, "concurrency": 1}

        cfg = BenchmarkConfig.model_validate(_minimal(phases=flat_load))

        assert any(p.name == "profiling" for p in cfg.phases)
        assert (
            next(p for p in cfg.phases if p.name == "profiling").type == "concurrency"
        )
        assert next(p for p in cfg.phases if p.name == "profiling").duration == 60.0

    def test_dict_of_phases_passthrough(self) -> None:
        multi = [
            {
                "name": "warmup",
                "type": "concurrency",
                "concurrency": 2,
                "requests": 10,
                "exclude_from_results": True,
            },
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 8,
                "requests": 100,
            },
        ]
        cfg = BenchmarkConfig.model_validate(_minimal(phases=multi))

        assert [p.name for p in cfg.phases] == ["warmup", "profiling"]

    def test_phase_names_injected(self) -> None:
        """Two phases with the same name 'profiling' must be rejected by uniqueness validator."""
        multi = [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 4,
                "requests": 50,
            },
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 16,
                "requests": 200,
            },
        ]
        with pytest.raises(ValidationError, match="duplicate phase name"):
            BenchmarkConfig.model_validate(_minimal(phases=multi))

    def test_single_phase_gets_default_name(self) -> None:
        flat_load = {"type": "concurrency", "requests": 10, "concurrency": 1}

        cfg = BenchmarkConfig.model_validate(_minimal(phases=flat_load))

        assert next(p for p in cfg.phases if p.name == "profiling").name == "profiling"

    @pytest.mark.parametrize(
        "phase_type,extra_fields",
        [
            ("concurrency", {"concurrency": 4, "requests": 50}),
            ("poisson", {"rate": 10.0, "requests": 50}),
            ("constant", {"rate": 5.0, "duration": 30}),
        ],
    )  # fmt: skip
    def test_single_phase_wrapping_works_for_all_types(
        self, phase_type: str, extra_fields: dict
    ) -> None:
        flat_load = {"type": phase_type, **extra_fields}
        cfg = BenchmarkConfig.model_validate(_minimal(phases=flat_load))

        assert any(p.name == "profiling" for p in cfg.phases)
        assert next(p for p in cfg.phases if p.name == "profiling").type == phase_type

    def test_load_not_dict_raises(self) -> None:
        with pytest.raises(Exception, match="phases must be a list"):
            BenchmarkConfig.model_validate(_minimal(phases="invalid"))

    def test_phase_value_not_dict_raises(self) -> None:
        with pytest.raises(Exception, match="phases must be a list"):
            BenchmarkConfig.model_validate(_minimal(phases={"bad": "not-a-dict"}))


# ============================================================
# Phase Flattening: warmup / profiling
# ============================================================


class TestPhaseFlattening:
    """Verify warmup/profiling top-level keys normalize into phases."""

    def test_profiling_only(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert any(p.name == "profiling" for p in cfg.phases)
        assert (
            next(p for p in cfg.phases if p.name == "profiling").type == "concurrency"
        )

    def test_warmup_and_profiling(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "warmup": {**_CONCURRENCY_PHASE, "requests": 5},
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert [p.name for p in cfg.phases] == ["warmup", "profiling"]
        assert (
            next(p for p in cfg.phases if p.name == "warmup").exclude_from_results
            is True
        )

    def test_warmup_auto_sets_exclude_from_results(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "warmup": _CONCURRENCY_PHASE,
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert (
            next(p for p in cfg.phases if p.name == "warmup").exclude_from_results
            is True
        )

    def test_warmup_user_excludeFromResults_camelcase_does_not_collide(self) -> None:
        """Regression: `warmup:` shorthand must not inject snake_case dup-key.

        Previously the normalizer always set `exclude_from_results` via
        ``setdefault``, which collided with a user-supplied camelCase
        ``excludeFromResults`` and tripped Pydantic's `extra="forbid"` on the
        phase model. Now the value is enforced by the phase validator instead
        of injected by the normalizer, so the collision cannot occur.
        """
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "warmup": {**_CONCURRENCY_PHASE, "excludeFromResults": True},
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert (
            next(p for p in cfg.phases if p.name == "warmup").exclude_from_results
            is True
        )

    def test_warmup_explicit_exclude_false_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "warmup": {**_CONCURRENCY_PHASE, "exclude_from_results": False},
            "profiling": _CONCURRENCY_PHASE,
        }
        with pytest.raises(ValueError, match="exclude_from_results must be True"):
            BenchmarkConfig.model_validate(data)

    def test_profiling_explicit_exclude_true_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "profiling": {**_CONCURRENCY_PHASE, "exclude_from_results": True},
        }
        with pytest.raises(ValueError, match="exclude_from_results must be False"):
            BenchmarkConfig.model_validate(data)

    def test_warmup_without_profiling_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "warmup": _CONCURRENCY_PHASE,
        }
        with pytest.raises(ValueError, match="'warmup' requires 'profiling'"):
            BenchmarkConfig.model_validate(data)

    def test_warmup_with_phases_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "warmup": _CONCURRENCY_PHASE,
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        with pytest.raises(Exception, match="'warmup' cannot be used with 'phases'"):
            BenchmarkConfig.model_validate(data)

    def test_profiling_with_phases_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "profiling": _CONCURRENCY_PHASE,
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        with pytest.raises(Exception, match="'profiling' cannot be used with 'phases'"):
            BenchmarkConfig.model_validate(data)

    def test_old_phases_form_still_works(self) -> None:
        data = _minimal()
        cfg = BenchmarkConfig.model_validate(data)

        assert any(p.name == "profiling" for p in cfg.phases)

    def test_warmup_preserves_execution_order(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [_DEFAULT_NAMED_DATASET],
            "profiling": _CONCURRENCY_PHASE,
            "warmup": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert [p.name for p in cfg.phases] == ["warmup", "profiling"]


class TestDatasetMutualExclusivity:
    """Verify dataset/datasets cannot both be present."""

    def test_dataset_and_datasets_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": _SYNTHETIC_DATASET,
            "datasets": [{"name": "other", **_SYNTHETIC_DATASET}],
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        with pytest.raises(Exception, match="'dataset' cannot be used with 'datasets'"):
            BenchmarkConfig.model_validate(data)


# ============================================================
# Dataset isl/osl Hoisting
# ============================================================


class TestIslOslHoisting:
    """Verify isl/osl at dataset level are hoisted into prompts."""

    def test_isl_osl_hoisted_in_singular_dataset(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": {"type": "synthetic", "entries": 100, "isl": 512, "osl": 128},
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets[0]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None
        assert ds.prompts.isl is not None
        assert ds.prompts.osl is not None

    def test_isl_osl_hoisted_in_named_datasets(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [
                {
                    "name": "a",
                    "type": "synthetic",
                    "entries": 50,
                    "isl": 256,
                    "osl": 64,
                },
            ],
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert cfg.datasets[0].prompts is not None
        assert cfg.datasets[0].prompts.isl is not None
        assert cfg.datasets[0].prompts.osl is not None

    def test_isl_only_hoisted(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": {"type": "synthetic", "entries": 100, "isl": 512},
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets[0]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None
        assert ds.prompts.isl is not None
        assert ds.prompts.osl is None

    def test_existing_prompts_form_unchanged(self) -> None:
        data = _minimal()
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets[0]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None

    def test_isl_osl_not_hoisted_on_file_dataset(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [
                {
                    "name": "default",
                    "type": "file",
                    "path": "/tmp/data.jsonl",
                    "isl": 512,
                }
            ],
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            BenchmarkConfig.model_validate(data)

    def test_isl_osl_not_hoisted_on_public_dataset(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": [
                {
                    "name": "profiling",
                    "type": "public",
                    "dataset": "sharegpt",
                    "isl": 512,
                }
            ],
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            BenchmarkConfig.model_validate(data)

    def test_isl_osl_default_type_synthetic(self) -> None:
        """When type is absent (defaults to synthetic), hoisting should work."""
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": {"entries": 100, "isl": 512, "osl": 128},
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets[0]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None

    def test_isl_osl_does_not_clobber_existing_prompts(self) -> None:
        """If prompts already exists, isl/osl at top level should merge."""
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": {
                "type": "synthetic",
                "entries": 100,
                "isl": 512,
                "prompts": {"osl": 64, "batch_size": 4},
            },
            "phases": [{"name": "profiling", **_CONCURRENCY_PHASE}],
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets[0]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None
        assert ds.prompts.isl is not None  # hoisted from top-level
        assert ds.prompts.osl is not None  # kept from existing prompts
        assert ds.prompts.batch_size == 4  # kept from existing prompts
