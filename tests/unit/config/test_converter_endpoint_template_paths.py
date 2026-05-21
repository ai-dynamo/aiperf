# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security regression tests for template-path reads in ``_converter_endpoint``.

Path-injection hardening for ``--extra-inputs payload_template=...``: the
converter must resolve the path through ``Path.resolve(strict=True)``,
refuse to follow symlinks, and read with explicit UTF-8 encoding. When any
safety check fails (missing path, symlink, non-regular file), the converter
falls back to treating the original string as a literal template body — the
pre-existing behavior for non-file inputs.

Covers both call sites that read a user-supplied template path:
``_endpoint_template_from_extra`` (line 33) and ``_endpoint_template_fallback``
(line 53).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from aiperf.config.flags._converter_endpoint import (
    _endpoint_template_fallback,
    _endpoint_template_from_extra,
    _safe_read_template_path,
)
from aiperf.plugin.enums import EndpointType


class TestEndpointTemplateFromExtraPathSafety:
    """``payload_template`` in ``extra_inputs`` must read files safely."""

    def test_regular_file_is_read_as_body(self, tmp_path: Path) -> None:
        template = tmp_path / "tmpl.json"
        template.write_text('{"hello": "world"}', encoding="utf-8")
        endpoint: dict = {}
        extra: dict = {"payload_template": str(template)}

        _endpoint_template_from_extra(endpoint, extra)

        assert endpoint["template"]["body"] == '{"hello": "world"}'

    def test_utf8_file_is_decoded_explicitly(self, tmp_path: Path) -> None:
        template = tmp_path / "tmpl.json"
        template.write_text('{"text": "café"}', encoding="utf-8")
        endpoint: dict = {}
        extra: dict = {"payload_template": str(template)}

        _endpoint_template_from_extra(endpoint, extra)

        assert endpoint["template"]["body"] == '{"text": "café"}'

    def test_symlink_is_rejected_and_string_used_as_literal_body(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "target.json"
        target.write_text('{"sensitive": "do-not-read"}', encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(target)
        endpoint: dict = {}
        extra: dict = {"payload_template": str(link)}

        _endpoint_template_from_extra(endpoint, extra)

        assert endpoint["template"]["body"] == str(link)

    def test_missing_path_falls_back_to_literal_body(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        endpoint: dict = {}
        extra: dict = {"payload_template": str(missing)}

        _endpoint_template_from_extra(endpoint, extra)

        assert endpoint["template"]["body"] == str(missing)

    def test_directory_path_falls_back_to_literal_body(self, tmp_path: Path) -> None:
        endpoint: dict = {}
        extra: dict = {"payload_template": str(tmp_path)}

        _endpoint_template_from_extra(endpoint, extra)

        assert endpoint["template"]["body"] == str(tmp_path)


class TestEndpointTemplateFallbackPathSafety:
    """Fallback path through ``endpoint['extra']`` must read files safely."""

    def test_regular_file_is_read_as_body(self, tmp_path: Path) -> None:
        template = tmp_path / "tmpl.json"
        template.write_text('{"hello": "world"}', encoding="utf-8")
        endpoint: dict = {
            "type": EndpointType.TEMPLATE,
            "extra": {"payload_template": str(template)},
        }

        _endpoint_template_fallback(endpoint)

        assert endpoint["template"]["body"] == '{"hello": "world"}'

    def test_symlink_is_rejected_and_string_used_as_literal_body(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "target.json"
        target.write_text('{"sensitive": "do-not-read"}', encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(target)
        endpoint: dict = {
            "type": EndpointType.TEMPLATE,
            "extra": {"payload_template": str(link)},
        }

        _endpoint_template_fallback(endpoint)

        assert endpoint["template"]["body"] == str(link)

    def test_missing_path_falls_back_to_literal_body(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        endpoint: dict = {
            "type": EndpointType.TEMPLATE,
            "extra": {"payload_template": str(missing)},
        }

        _endpoint_template_fallback(endpoint)

        assert endpoint["template"]["body"] == str(missing)

    @pytest.mark.parametrize(
        "ep_type",
        [
            pytest.param(EndpointType.CHAT, id="chat"),
            pytest.param(EndpointType.COMPLETIONS, id="completions"),
        ],
    )
    def test_non_template_endpoints_skip_fallback(
        self, tmp_path: Path, ep_type: EndpointType
    ) -> None:
        template = tmp_path / "tmpl.json"
        template.write_text("body", encoding="utf-8")
        endpoint: dict = {
            "type": ep_type,
            "extra": {"payload_template": str(template)},
        }

        _endpoint_template_fallback(endpoint)

        assert "template" not in endpoint


class TestSafeReadTemplatePathReadFailures:
    """``_safe_read_template_path`` must swallow ``OSError`` from the read step.

    ``resolve(strict=True)`` already exercises the ``OSError`` branch via
    ``FileNotFoundError`` for missing paths. The remaining uncovered branch is
    a read failure on a file that passed ``is_file()`` — e.g., permission
    denied, or a TOCTOU delete between the stat and the open.
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX chmod semantics; Windows ACLs do not block owner reads via chmod",
    )
    @pytest.mark.skipif(
        os.geteuid() == 0 if hasattr(os, "geteuid") else False,
        reason="root bypasses file mode permission checks",
    )
    def test_unreadable_file_returns_none(self, tmp_path: Path) -> None:
        unreadable = tmp_path / "locked.json"
        unreadable.write_text('{"hello": "world"}', encoding="utf-8")
        unreadable.chmod(0o000)
        try:
            assert _safe_read_template_path(str(unreadable)) is None
        finally:
            unreadable.chmod(0o644)


class TestSafeReadTemplatePathConstructionFailures:
    """``_safe_read_template_path`` must reject inputs that fail ``Path()`` construction.

    Defensive against upstream parsers that hand the helper a non-string value
    (e.g., ``--extra-inputs`` JSON parsed to an int/None for ``payload_template``).
    ``Path(non_str)`` raises ``TypeError`` and certain invalid string contents raise
    ``ValueError``; both must collapse to ``None`` rather than propagate.
    """

    @pytest.mark.parametrize(
        "bad_input",
        [
            pytest.param(42, id="int"),
            pytest.param(None, id="none"),
            pytest.param(["a", "b"], id="list"),
            pytest.param({"x": 1}, id="dict"),
        ],
    )
    def test_non_string_input_returns_none(self, bad_input: object) -> None:
        # ``ts`` is typed ``str`` but the helper guards against parser
        # misbehavior that smuggles a non-str through ``dict.get``.
        assert _safe_read_template_path(bad_input) is None  # type: ignore[arg-type]
