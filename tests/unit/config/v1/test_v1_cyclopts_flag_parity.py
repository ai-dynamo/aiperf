# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify UserConfig + ServiceConfig produce a CLI flag set >= the current CLIModel.

This guards against flags accidentally lost when restoring v1 from origin/main.

Approach A (programmatic walk) is used: cyclopts exposes
`App.assemble_argument_collection()`, which returns an `ArgumentCollection`
of `Argument` objects whose `.names` tuple holds every long/short flag the
parser would accept. This is more reliable than parsing `--help` output,
which is formatted for humans (line-wrapped, group-headered) and would
require brittle text scraping.
"""

from cyclopts import App

from aiperf.config.cli_model import CLIModel
from aiperf.config.v1 import ServiceConfig, UserConfig


def _flags_for(*, name: str, **typed_params: type) -> set[str]:
    """Return the long-form CLI flag names cyclopts synthesizes for the given Pydantic model parameters.

    `typed_params` maps the parameter-name (used by cyclopts to form the
    "wrapper" flag, e.g. `--cli-model`) to the Pydantic class. We strip out
    those wrapper flags from the result so the comparison is between the
    field-derived flags only.
    """
    app = App(name=name)

    # Build a function whose kwargs are typed with the requested Pydantic
    # models, then register it as the default command. Cyclopts walks the
    # Pydantic fields recursively to synthesize per-field flags.
    param_names = list(typed_params.keys())
    annotations = dict(typed_params)
    annotations["return"] = type(None)

    def _cmd(**_kwargs: object) -> None:  # pragma: no cover - never invoked
        return None

    _cmd.__annotations__ = annotations
    # Force kw-only so cyclopts treats them as flags, not positionals.
    import inspect

    params = [
        inspect.Parameter(
            n,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=typed_params[n],
        )
        for n in param_names
    ]
    _cmd.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=params, return_annotation=type(None)
    )

    app.default(_cmd)

    # Wrapper flags are the parameter names with underscores -> dashes,
    # prefixed with `--`. These are the per-Pydantic-arg roots that cyclopts
    # adds and we don't want to count in field-flag parity.
    wrapper_flags = {f"--{n.replace('_', '-')}" for n in param_names}

    flags: set[str] = set()
    for arg in app.assemble_argument_collection():
        for n in arg.names:
            if not n.startswith("--"):
                continue  # ignore short aliases like `-m`
            if n in wrapper_flags:
                continue
            # Dotted flags (e.g. `--cli-model.benchmark-id`) are emitted by
            # cyclopts for `parse=False` / generated fields that don't take a
            # plain top-level form. They are not user-facing CLI flags.
            if "." in n:
                continue
            flags.add(n)
    return flags


def test_user_config_flag_parity_with_cli_model() -> None:
    v1_flags = _flags_for(name="v1", user=UserConfig, service=ServiceConfig)
    flat_flags = _flags_for(name="flat", cli_model=CLIModel)

    missing = flat_flags - v1_flags
    assert not missing, (
        f"v1 restoration is missing {len(missing)} flags from CLIModel: "
        f"{sorted(missing)}"
    )
