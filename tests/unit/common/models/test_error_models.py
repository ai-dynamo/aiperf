# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ErrorDetails cause chain building."""

import pytest

from aiperf.common.models.error_models import ErrorDetails


class TestBuildCauseChain:
    """Tests for ErrorDetails._build_cause_chain."""

    def test_explicit_cause_chain(self):
        """Explicit chaining via 'raise X from Y' produces correct chain."""
        try:
            try:
                raise ValueError("root")
            except ValueError as e:
                raise RuntimeError("wrapper") from e
        except RuntimeError as e:
            chain = ErrorDetails._build_cause_chain(e)

        assert chain == ["RuntimeError", "ValueError"]

    def test_implicit_context_chain(self):
        """Implicit chaining (raise inside except) follows __context__."""
        try:
            try:
                raise ValueError("root")
            except ValueError:
                raise RuntimeError("wrapper")  # noqa: B904
        except RuntimeError as e:
            chain = ErrorDetails._build_cause_chain(e)

        assert chain == ["RuntimeError", "ValueError"]

    def test_mixed_cause_and_context(self):
        """Chain with both __cause__ and __context__ links."""
        try:
            try:
                try:
                    raise KeyError("deep root")
                except KeyError:
                    raise ValueError("middle")  # noqa: B904 — implicit context
            except ValueError as e:
                raise RuntimeError("top") from e  # explicit cause
        except RuntimeError as e:
            chain = ErrorDetails._build_cause_chain(e)

        assert chain == ["RuntimeError", "ValueError", "KeyError"]

    def test_suppressed_context_stops_chain(self):
        """'raise X from None' suppresses context and stops chain."""
        try:
            try:
                raise ValueError("root")
            except ValueError:
                raise RuntimeError("wrapper") from None
        except RuntimeError as e:
            chain = ErrorDetails._build_cause_chain(e)

        assert chain == ["RuntimeError"]

    def test_single_exception(self):
        """Single exception with no chaining."""
        try:
            raise ValueError("only")
        except ValueError as e:
            chain = ErrorDetails._build_cause_chain(e)

        assert chain == ["ValueError"]

    def test_none_returns_none(self):
        """None input returns None (edge case)."""
        assert ErrorDetails._build_cause_chain(None) is None

    def test_tokenizer_error_scenario(self):
        """Simulates transformers wrapping a HF Hub error without 'from'.

        This is the exact scenario that was failing: transformers raises OSError
        inside an except block (implicit __context__), then our code wraps it
        in TokenizerError with explicit 'from'.
        """

        class GatedRepoError(Exception):
            pass

        class TokenizerError(Exception):
            pass

        try:
            try:
                try:
                    raise GatedRepoError("gated")
                except GatedRepoError:
                    raise OSError("Can't load tokenizer")  # noqa: B904
            except OSError as e:
                raise TokenizerError("Failed") from e
        except TokenizerError as e:
            chain = ErrorDetails._build_cause_chain(e)

        assert chain == ["TokenizerError", "OSError", "GatedRepoError"]

    @pytest.mark.parametrize(
        ("chain_builder", "expected"),
        [
            pytest.param(
                lambda: (_ for _ in ()).throw(ValueError("solo")),
                ["ValueError"],
                id="no_chain",
            ),
        ],
    )
    def test_parametrized_chains(self, chain_builder, expected):
        """Parametrized chain tests."""
        try:
            chain_builder()
        except Exception as e:
            chain = ErrorDetails._build_cause_chain(e)
        assert chain == expected
