# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSTI hardening: user_files content is rendered in a jinja2 sandbox.

``artifacts.user_files`` content originates from the same attacker-controlled
CRD spec and is rendered (in the operator pod) at run start. A bare
``jinja2.Environment`` made that a server-side template injection -> RCE gadget.
These tests pin the sandbox: escape gadgets raise (wrapped as UserFileError)
instead of executing, while legitimate ``{{ variable }}`` content still renders.
"""

from __future__ import annotations

import pytest
from jinja2.sandbox import SandboxedEnvironment
from pytest import param

from aiperf.config.user_files import (
    _USER_FILES_ENV,
    UserFile,
    UserFileError,
    materialize_user_files,
)

SSTI_GADGETS = [
    param(
        "{{ cycler.__init__.__globals__.os.popen('id').read() }}",
        id="cycler-os-popen",
    ),
    param("{{ ''.__class__.__mro__ }}", id="str-class-mro"),
    param("{{ self.__init__.__globals__ }}", id="self-init-globals"),
    param(
        "{{ cycler.__init__.__globals__.__builtins__['__import__']('os').system('id') }}",
        id="builtins-import-os-system",
    ),
]


def test_user_files_env_is_sandboxed() -> None:
    """The user_files env must be a sandbox subclass, not a bare Environment."""
    assert isinstance(_USER_FILES_ENV, SandboxedEnvironment)


@pytest.mark.parametrize("gadget", SSTI_GADGETS)
def test_user_files_ssti_gadget_blocked(gadget: str, tmp_path) -> None:
    """A gadget in user_files content raises UserFileError, never executes."""
    entry = UserFile(path="out.txt", format="text", content=gadget)
    with pytest.raises(UserFileError):
        materialize_user_files([entry], tmp_path, context={})


def test_user_files_ssti_side_effect_never_runs(tmp_path) -> None:
    """Gold-standard 'did not execute': the sentinel file must not be created."""
    sentinel = tmp_path / "aiperf_userfiles_pwned"
    gadget = (
        "{{ cycler.__init__.__globals__.os.popen('touch "
        + str(sentinel)
        + "').read() }}"
    )
    entry = UserFile(path="out.txt", format="text", content=gadget)
    with pytest.raises(UserFileError):
        materialize_user_files([entry], tmp_path, context={})
    assert not sentinel.exists(), "user_files SSTI gadget executed a shell command"


def test_user_files_legit_content_still_renders(tmp_path) -> None:
    """Sandbox must NOT break legitimate user_files templating."""
    entry = UserFile(
        path="meta.txt",
        format="text",
        content="model={{ model }} epoch={{ epoch }} n={{ n * 2 }}\n",
    )
    materialize_user_files(
        [entry],
        tmp_path,
        context={"model": "mock-llm", "epoch": "1714000000", "n": 21},
    )
    assert (
        tmp_path / "meta.txt"
    ).read_text() == "model=mock-llm epoch=1714000000 n=42\n"
