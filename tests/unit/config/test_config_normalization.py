# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for before-validators that normalize shorthand config input.

Focuses on:
- models: string / list[str] / singular "model" key -> ModelsAdvanced
- datasets: singular "dataset" key -> wrapped under "default", composed detection, default type
- load: single phase dict (has "type") -> wrapped under "default", phase name injection
"""

from __future__ import annotations

import pytest

from aiperf.config.config import BenchmarkConfig
from aiperf.config.dataset import ComposedDataset, SyntheticDataset
from aiperf.config.models import ModelsAdvanced

_ENDPOINT = {"urls": ["http://localhost:8000/v1/chat/completions"]}
_SYNTHETIC_DATASET = {
    "type": "synthetic",
    "entries": 100,
    "prompts": {"isl": 128, "osl": 64},
}
_CONCURRENCY_PHASE = {"type": "concurrency", "requests": 10, "concurrency": 1}


def _minimal(**overrides: object) -> dict:
    """Minimal valid BenchmarkConfig dict with overrides."""
    base: dict = {
        "models": ["m"],
        "endpoint": _ENDPOINT,
        "datasets": {"default": _SYNTHETIC_DATASET},
        "phases": {"default": _CONCURRENCY_PHASE},
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

    def test_singular_model_ignored_when_models_present(self) -> None:
        """Normalizer skips "model" when "models" exists; leftover key triggers extra="forbid"."""
        data = _minimal(models=["keep-me"])
        data["model"] = "ignore-me"

        with pytest.raises(Exception, match="Extra inputs are not permitted"):
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

        assert "default" in cfg.datasets
        assert isinstance(cfg.datasets["default"], SyntheticDataset)

    def test_composed_dataset_with_explicit_type(self) -> None:
        composed_dict = {
            "type": "composed",
            "source": {
                "type": "file",
                "path": "/tmp/data.jsonl",
            },
            "augment": {
                "osl": {"mean": 128, "stddev": 20},
            },
        }
        cfg = BenchmarkConfig.model_validate(
            _minimal(datasets={"mixed": composed_dict})
        )

        assert isinstance(cfg.datasets["mixed"], ComposedDataset)

    def test_composed_without_explicit_type_needs_type_field(self) -> None:
        """source+augment without type field fails: discriminated union requires tag."""
        composed_dict = {
            "source": {
                "type": "file",
                "path": "/tmp/data.jsonl",
            },
            "augment": {
                "osl": {"mean": 128, "stddev": 20},
            },
        }
        with pytest.raises(Exception, match="Unable to extract tag"):
            BenchmarkConfig.model_validate(_minimal(datasets={"mixed": composed_dict}))

    def test_default_type_synthetic(self) -> None:
        no_type = {"entries": 50, "prompts": {"isl": 64}}
        cfg = BenchmarkConfig.model_validate(_minimal(datasets={"gen": no_type}))

        assert isinstance(cfg.datasets["gen"], SyntheticDataset)

    def test_explicit_type_preserved(self) -> None:
        cfg = BenchmarkConfig.model_validate(
            _minimal(
                datasets={
                    "trace": {
                        "type": "file",
                        "path": "/tmp/trace.jsonl",
                        "format": "mooncake_trace",
                    }
                }
            )
        )

        assert cfg.datasets["trace"].type == "file"


# ============================================================
# Load Normalization
# ============================================================


class TestLoadNormalization:
    """Verify normalize_before_validation and parse_load for load section."""

    def test_single_phase_wrapped_with_default_key(self) -> None:
        flat_load = {"type": "concurrency", "duration": 60, "concurrency": 1}

        cfg = BenchmarkConfig.model_validate(_minimal(phases=flat_load))

        assert "default" in cfg.phases
        assert cfg.phases["default"].type == "concurrency"
        assert cfg.phases["default"].duration == 60.0

    def test_dict_of_phases_passthrough(self) -> None:
        multi = {
            "warmup": {
                "type": "concurrency",
                "concurrency": 2,
                "requests": 10,
                "exclude_from_results": True,
            },
            "main": {
                "type": "concurrency",
                "concurrency": 8,
                "requests": 100,
            },
        }
        cfg = BenchmarkConfig.model_validate(_minimal(phases=multi))

        assert list(cfg.phases.keys()) == ["warmup", "main"]

    def test_phase_names_injected(self) -> None:
        multi = {
            "ramp_up": {
                "type": "concurrency",
                "concurrency": 4,
                "requests": 50,
                "exclude_from_results": True,
            },
            "profiling": {
                "type": "concurrency",
                "concurrency": 16,
                "requests": 200,
            },
        }
        cfg = BenchmarkConfig.model_validate(_minimal(phases=multi))

        assert cfg.phases["ramp_up"].name == "ramp_up"
        assert cfg.phases["profiling"].name == "profiling"

    def test_single_phase_gets_default_name(self) -> None:
        flat_load = {"type": "concurrency", "requests": 10, "concurrency": 1}

        cfg = BenchmarkConfig.model_validate(_minimal(phases=flat_load))

        assert cfg.phases["default"].name == "default"

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

        assert "default" in cfg.phases
        assert cfg.phases["default"].type == phase_type

    def test_load_not_dict_raises(self) -> None:
        with pytest.raises(Exception, match="phases must be a dictionary"):
            BenchmarkConfig.model_validate(_minimal(phases="invalid"))

    def test_phase_value_not_dict_raises(self) -> None:
        with pytest.raises(Exception, match="must be a dictionary"):
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
            "datasets": {"default": _SYNTHETIC_DATASET},
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert "profiling" in cfg.phases
        assert cfg.phases["profiling"].type == "concurrency"

    def test_warmup_and_profiling(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {"default": _SYNTHETIC_DATASET},
            "warmup": {**_CONCURRENCY_PHASE, "requests": 5},
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert list(cfg.phases.keys()) == ["warmup", "profiling"]
        assert cfg.phases["warmup"].exclude_from_results is True

    def test_warmup_auto_sets_exclude_from_results(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {"default": _SYNTHETIC_DATASET},
            "warmup": _CONCURRENCY_PHASE,
            "profiling": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert cfg.phases["warmup"].exclude_from_results is True

    def test_warmup_without_profiling_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {"default": _SYNTHETIC_DATASET},
            "warmup": _CONCURRENCY_PHASE,
        }
        with pytest.raises(ValueError, match="'warmup' requires 'profiling'"):
            BenchmarkConfig.model_validate(data)

    def test_warmup_with_phases_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {"default": _SYNTHETIC_DATASET},
            "warmup": _CONCURRENCY_PHASE,
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        with pytest.raises(Exception, match="'warmup' cannot be used with 'phases'"):
            BenchmarkConfig.model_validate(data)

    def test_profiling_with_phases_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {"default": _SYNTHETIC_DATASET},
            "profiling": _CONCURRENCY_PHASE,
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        with pytest.raises(Exception, match="'profiling' cannot be used with 'phases'"):
            BenchmarkConfig.model_validate(data)

    def test_old_phases_form_still_works(self) -> None:
        data = _minimal()
        cfg = BenchmarkConfig.model_validate(data)

        assert "default" in cfg.phases

    def test_warmup_preserves_execution_order(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {"default": _SYNTHETIC_DATASET},
            "profiling": _CONCURRENCY_PHASE,
            "warmup": _CONCURRENCY_PHASE,
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert list(cfg.phases.keys()) == ["warmup", "profiling"]


class TestDatasetMutualExclusivity:
    """Verify dataset/datasets cannot both be present."""

    def test_dataset_and_datasets_rejected(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": _SYNTHETIC_DATASET,
            "datasets": {"other": _SYNTHETIC_DATASET},
            "phases": {"default": _CONCURRENCY_PHASE},
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
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets["default"]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None
        assert ds.prompts.isl is not None
        assert ds.prompts.osl is not None

    def test_isl_osl_hoisted_in_named_datasets(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {
                "a": {"type": "synthetic", "entries": 50, "isl": 256, "osl": 64},
                "b": {
                    "type": "synthetic",
                    "entries": 50,
                    "isl": {"mean": 512, "stddev": 50},
                    "osl": 128,
                },
            },
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        cfg = BenchmarkConfig.model_validate(data)

        assert cfg.datasets["a"].prompts is not None
        assert cfg.datasets["b"].prompts is not None

    def test_isl_only_hoisted(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": {"type": "synthetic", "entries": 100, "isl": 512},
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets["default"]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None
        assert ds.prompts.isl is not None
        assert ds.prompts.osl is None

    def test_existing_prompts_form_unchanged(self) -> None:
        data = _minimal()
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets["default"]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None

    def test_isl_osl_not_hoisted_on_file_dataset(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {
                "default": {"type": "file", "path": "/tmp/data.jsonl", "isl": 512},
            },
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            BenchmarkConfig.model_validate(data)

    def test_isl_osl_not_hoisted_on_public_dataset(self) -> None:
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": {
                "default": {"type": "public", "name": "sharegpt", "isl": 512},
            },
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            BenchmarkConfig.model_validate(data)

    def test_isl_osl_default_type_synthetic(self) -> None:
        """When type is absent (defaults to synthetic), hoisting should work."""
        data = {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "dataset": {"entries": 100, "isl": 512, "osl": 128},
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets["default"]
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
            "phases": {"default": _CONCURRENCY_PHASE},
        }
        cfg = BenchmarkConfig.model_validate(data)

        ds = cfg.datasets["default"]
        assert isinstance(ds, SyntheticDataset)
        assert ds.prompts is not None
        assert ds.prompts.isl is not None  # hoisted from top-level
        assert ds.prompts.osl is not None  # kept from existing prompts
        assert ds.prompts.batch_size == 4  # kept from existing prompts
