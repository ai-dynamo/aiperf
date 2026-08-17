# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base configuration model with camelCase alias support.

All user-facing config models inherit BaseConfig so that:
- Serialization (model_dump / YAML / JSON) uses camelCase keys
- Deserialization accepts both camelCase and snake_case input
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def hide_from_unset_dumps(model: BaseModel, field_name: str) -> None:
    """Drop ``field_name`` from ``model_fields_set`` so ``exclude_unset`` omits it.

    For internal provenance flags (``streaming_explicitly_set``,
    ``target_explicitly_set``, ``concurrency_explicitly_set``) which are
    deliberately SERIALIZED but must not be user-facing.

    They cannot use ``exclude=True``: the sweep orchestrator round-trips every
    run through ``local_executor._prepare_run_artifacts``
    (``model_dump(exclude_none=True)``) -> ``subprocess_runner``
    (``model_validate``), and ``model_fields_set`` is uninformative on the far
    side because every dumped key returns marked "set". Dropping the flag there
    would let a value the author never chose read as explicitly authored.

    But their validators ASSIGN the flag, and pydantic marks an assigned field
    "set" -- so the exporters' ``exclude_unset=True`` kept it and the keys
    surfaced in ``profile_export_aiperf.json`` under ``input_config``. The two
    dump shapes are what separates the concerns::

        exporters       model_dump(mode="json", exclude_unset=True, exclude_none=True)
        sweep boundary  model_dump(mode="json", exclude_none=True)

    Only the exporters pass ``exclude_unset``, so clearing the set-marker hides
    the flag from every export while the sweep dump still carries it.

    Called UNCONDITIONALLY, after the validator settles the value -- including
    when the flag arrived as an incoming key. A sweep cell's subprocess writes
    its own ``profile_export_aiperf.json``, and there the flag IS a supplied
    key; discarding only computed values would leak it on exactly that path.
    Clearing the marker never changes the value, only whether ``exclude_unset``
    emits it.

    Args:
        model: The model instance whose set-marker should be cleared.
        field_name: Field to hide from ``exclude_unset`` dumps.
    """
    model.__pydantic_fields_set__.discard(field_name)


class BaseConfig(BaseModel):
    """Base for all AIPerf configuration models.

    Provides camelCase alias generation for K8s CRD compatibility
    while keeping Python field names as snake_case.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
