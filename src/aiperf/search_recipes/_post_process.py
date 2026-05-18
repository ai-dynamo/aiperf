# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Leaf module hosting :class:`PostProcessSpec`.

Lives in its own module so ``aiperf.config.sweep`` (which carries a
``post_process: PostProcessSpec | None`` field on ``_SweepBase``) can import
the type without pulling in the rest of ``aiperf.search_recipes._base`` --
``_base`` imports ``aiperf.config.config.BenchmarkConfig``, which itself
re-exports the sweep types, so the import would cycle. Inheriting from
``pydantic.BaseModel`` directly (rather than ``aiperf.config.base.BaseConfig``)
is also load-bearing: ``aiperf.config.base`` triggers the ``aiperf.config``
package init, which loads ``aiperf.config.resolution.plan`` -> ``aiperf.config.config``
-> ``aiperf.config.sweep``, looping back through this module while it's still
initializing.

``aiperf.search_recipes._base`` re-exports ``PostProcessSpec`` for callers
that already import it from there.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["PostProcessSpec"]


class PostProcessSpec(BaseModel):
    """Post-aggregation hook spec emitted by a search recipe.

    Resolved through the ``search_recipe_post_process`` plugin category and
    invoked by ``aggregate_sweep_and_export`` after per-variation aggregation.

    Example: ``PostProcessSpec(handler="ttft_sla_curve", params={"sla_ms": 200},
    output_filename="ttft_sla_curve.json")``.
    """

    model_config = ConfigDict(extra="forbid")

    handler: str = Field(
        description=(
            "Registered post-process handler name. Looked up in the "
            "``search_recipe_post_process`` plugin category."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form parameters forwarded to the handler. Shape is handler-defined; "
            "Any is intentional because handlers ingest dynamic recipe-derived inputs."
        ),
    )
    output_filename: str = Field(
        description=(
            "Filename (relative to ``sweep_aggregate/`` under the artifact dir) "
            "to write the handler's output to. Must end in ``.json`` and contain "
            "no path separators -- the hook joins it with ``sweep_aggregate/`` "
            "and refuses to escape that directory. NUL bytes and dot-only stems "
            "are also rejected."
        ),
    )

    @model_validator(mode="after")
    def _check_handler(self) -> PostProcessSpec:
        if not self.handler.strip():
            raise ValueError(
                "PostProcessSpec.handler must be a non-empty, non-whitespace "
                "registered handler name."
            )
        return self

    @model_validator(mode="after")
    def _check_output_filename(self) -> PostProcessSpec:
        # Reject path separators outright so a recipe can't write to ``..`` or
        # an absolute path; the hook joins this filename with ``sweep_aggregate/``
        # and we want filename-only there.
        if "/" in self.output_filename or "\\" in self.output_filename:
            raise ValueError(
                f"PostProcessSpec.output_filename must be filename-only "
                f"(no path separators), got {self.output_filename!r}."
            )
        if "\x00" in self.output_filename:
            raise ValueError(
                f"PostProcessSpec.output_filename must not contain NUL bytes, "
                f"got {self.output_filename!r}."
            )
        if not self.output_filename.endswith(".json"):
            raise ValueError(
                f"PostProcessSpec.output_filename must end in '.json' "
                f"(handlers emit JSON artifacts), got {self.output_filename!r}."
            )
        # Reject "leading-dot only" filenames: '.json', '..json', '...json'
        # produce hidden / parent-resembling files that surprise users even
        # though they don't traverse out of sweep_aggregate/.
        stem = self.output_filename[: -len(".json")]
        if stem == "" or set(stem) == {"."}:
            raise ValueError(
                f"PostProcessSpec.output_filename must have a non-dot stem "
                f"before '.json' (got {self.output_filename!r}); hidden / "
                f"dot-only filenames are confusing on disk."
            )
        return self
