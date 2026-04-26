# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify _CLI_GROUP class attrs survive model_rebuild() under PEP 563.

A bare class-body assignment ``_CLI_GROUP = Groups.X`` is consumed by Pydantic's
field machinery; under ``from __future__ import annotations`` (PEP 563) the
stringified ``Annotated[..., CLIParameter(group=_CLI_GROUP)]`` annotations can
no longer resolve ``_CLI_GROUP`` when ``model_rebuild()`` re-evaluates them,
producing ``NameError: name '_CLI_GROUP' is not defined``. Annotating as
``ClassVar[Group]`` keeps it as a true class attribute that the closure can see.

This regression test calls ``model_rebuild()`` on every v1 nested class so any
future addition of ``from __future__ import annotations`` to one of these
modules (without the ClassVar annotation) is caught immediately.
"""

import pytest

from aiperf.config.v1._accuracy import AccuracyConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.config.v1._input import InputConfig
from aiperf.config.v1._loadgen import LoadGeneratorConfig
from aiperf.config.v1._output import OutputConfig
from aiperf.config.v1._tokenizer import TokenizerConfig
from aiperf.config.v1._workers import WorkersConfig
from aiperf.config.v1._zmq import ZMQDualBindConfig, ZMQIPCConfig, ZMQTCPConfig


@pytest.mark.parametrize(
    "cls",
    [
        EndpointConfig,
        InputConfig,
        LoadGeneratorConfig,
        OutputConfig,
        TokenizerConfig,
        AccuracyConfig,
        WorkersConfig,
        ZMQTCPConfig,
        ZMQIPCConfig,
        ZMQDualBindConfig,
    ],
)
def test_model_rebuild_does_not_explode(cls):
    """Calling model_rebuild() on each v1 nested class must not raise NameError."""
    cls.model_rebuild()  # must not raise
