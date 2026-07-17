# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python code-template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _DB_TABLES,
    _DECORATORS,
    _ERROR_MESSAGES,
    _HTTP_ROUTES,
    _METHODS,
    _MODULES,
    _TYPES,
    _VARS,
)


class _PythonMixin:
    def _gen_python_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_python_class,
                self._gen_python_functions,
                self._gen_python_test,
                self._gen_python_http_handler,
                self._gen_python_data_model,
            ]
        )()

    def _gen_python_class(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        t1, t2 = r.sample(_TYPES, 2)
        dec = r.choice(_DECORATORS)
        imp_mod = r.choice(_MODULES)
        imp_cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {mod}
from {mod}.{imp_mod} import {imp_cls}


class {cls}:
    \"\"\"Handles {m1} operations for {mod}.\"\"\"

    _default_{v3} = 64

    def __init__(self, {v1}: {t1}, {v2}: {t2} = None):
        self._{v1} = {v1}
        self._{v2} = {v2}
        self._{v3} = self._default_{v3}
        self._initialized = False

    {dec}
    async def {m1}(self, {v1}: {t1}) -> {t2}:
        if not self._initialized:
            raise RuntimeError("{cls} not initialized")
        {v2} = await self._{m2}({v1})
        return {v2}

    async def _{m2}(self, {v1}: {t1}) -> {t2}:
        try:
            {v2} = {mod}.{m2}({v1})
            return {v2}
        except Exception as e:
            raise ValueError("{err}") from e

    def {m3}(self) -> None:
        self._initialized = True
        self._{v3} = 0
"""

    def _gen_python_functions(self) -> str:
        r = self._template_rng
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        t1, t2, t3 = r.sample(_TYPES, 3)
        mod = r.choice(_MODULES)
        imp_mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from {mod}.{imp_mod} import {cls}

logger = logging.getLogger(__name__)


async def {m1}({v1}: {t1}, {v2}: {t2} | None = None) -> {t3}:
    async with _acquire_{v3}({v1}) as {v3}:
        {v2} = await {cls}().{m2}({v3})
        return [{v2} for _ in range(10) if {v2} is not None]


@asynccontextmanager
async def _acquire_{v3}({v1}: {t1}) -> AsyncIterator[{t2}]:
    {v3} = {mod}.{m3}({v1})
    try:
        yield {v3}
    finally:
        await {v3}.close()


def {m2}_sync({v1}: {t1}, *, max_retries: int = 3) -> {t2}:
    for attempt in range(max_retries):
        try:
            return {mod}.{m2}({v1})
        except RuntimeError:
            if attempt == max_retries - 1:
                raise
            logger.warning("{err}, attempt %d", attempt + 1)
    raise AssertionError("unreachable")
"""

    def _gen_python_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import pytest
from unittest.mock import AsyncMock, patch

from {mod} import {cls}


class Test{cls}:
    @pytest.fixture
    def instance(self):
        return {cls}({v1}="test_value")

    @pytest.mark.asyncio
    async def test_{m1}_returns_expected(self, instance):
        instance._{m2} = AsyncMock(return_value=42)
        result = await instance.{m1}()
        assert result == 42
        instance._{m2}.assert_awaited_once()

    @pytest.mark.parametrize("{v1}", ["alpha", "beta", "gamma"])
    def test_{m2}_with_values(self, instance, {v1}):
        instance._{v1} = {v1}
        result = instance.{m2}()
        assert result is not None

    @pytest.mark.asyncio
    async def test_{m3}_raises_on_{v2}(self, instance):
        with pytest.raises(ValueError, match="{err}"):
            await instance.{m3}(None)

    @pytest.mark.asyncio
    async def test_{m1}_with_mock_dependency(self, instance):
        with patch("{mod}.{m2}") as mock:
            mock.return_value = {{{{"key": "{v2}"}}}}\n            result = await instance.{m1}()
            assert "{v2}" in str(result)
"""

    def _gen_python_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        route = r.choice(_HTTP_ROUTES)
        table = r.choice(_DB_TABLES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from {mod}.{cls.lower()} import {cls}

router = APIRouter(prefix="{route}", tags=["{mod}"])


class {cls}Request(BaseModel):
    {v1}: str = Field(description="Primary {v1} identifier")
    {v2}: int = Field(default=10, ge=1, le=100, description="Page size")
    {v3}: str | None = Field(default=None, description="Optional filter")


class {cls}Response(BaseModel):
    items: list[dict] = Field(description="Result items from {table}")
    total: int = Field(description="Total count")
    page: int = Field(description="Current page number")


@router.post("/", response_model={cls}Response, status_code=201)
async def {m1}(
    body: {cls}Request,
    svc: {cls} = Depends(),
) -> {cls}Response:
    try:
        items = await svc.{m1}(body.{v1}, page_size=body.{v2})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {cls}Response(items=items, total=len(items), page=1)


@router.get("/{{{{{v1}}}}}")
async def {m2}({v1}: str, svc: {cls} = Depends()) -> dict:
    result = await svc.{m2}({v1})
    if result is None:
        raise HTTPException(status_code=404, detail="{err}")
    return {{"status": "ok", "data": result}}
"""

    def _gen_python_data_model(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        v1, v2, v3, v4 = r.sample(_VARS, 4)
        m1 = r.choice(_METHODS)
        table = r.choice(_DB_TABLES)

        return f"""\
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class {cls}Status(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class {cls}Config(BaseModel):
    {v1}: str = Field(description="{cls} {v1} identifier")
    {v2}: int = Field(default=0, ge=0, description="Current {v2} count")
    {v3}: float = Field(default=1.0, gt=0, description="Rate limit for {m1}")
    status: {cls}Status = Field(default={cls}Status.PENDING, description="Lifecycle status")
    {v4}: dict[str, str] = Field(default_factory=dict, description="Arbitrary {v4}")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    source_table: str = Field(default="{table}", description="Backing store table")

    @field_validator("{v1}")
    @classmethod
    def _validate_{v1}(cls, v: str) -> str:
        if not v or len(v) > 256:
            raise ValueError("{v1} must be 1-256 characters")
        return v.strip()

    @field_validator("{v3}")
    @classmethod
    def _validate_{v3}(cls, v: float) -> float:
        if v > 10_000:
            raise ValueError("{v3} exceeds max rate")
        return v

    def {m1}(self) -> bool:
        return self.status == {cls}Status.ACTIVE and self.{v2} > 0
"""
