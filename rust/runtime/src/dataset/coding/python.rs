// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python code-template renderers.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;

/// Dispatch across the Python structural variants.
pub(super) fn render(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    match r.index(5)? {
        0 => class(r),
        1 => functions(r),
        2 => test(r),
        3 => http_handler(r),
        _ => data_model(r),
    }
}

fn class(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let mod_ = r.pick(MODULES)?;
    let m = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (m[0], m[1], m[2]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let t = r.sample(TYPES, 2)?;
    let (t1, t2) = (t[0], t[1]);
    let dec = r.pick(DECORATORS)?;
    let imp_mod = r.pick(MODULES)?;
    let imp_cls = r.pick(CLASSES)?;
    let err = r.pick(ERRORS)?;

    Ok(format!(
        r#"import {mod_}
from {mod_}.{imp_mod} import {imp_cls}


class {cls}:
    """Handles {m1} operations for {mod_}."""

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
            {v2} = {mod_}.{m2}({v1})
            return {v2}
        except Exception as e:
            raise ValueError("{err}") from e

    def {m3}(self) -> None:
        self._initialized = True
        self._{v3} = 0
"#
    ))
}

fn functions(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let m = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (m[0], m[1], m[2]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let t = r.sample(TYPES, 3)?;
    let (t1, t2, t3) = (t[0], t[1], t[2]);
    let mod_ = r.pick(MODULES)?;
    let imp_mod = r.pick(MODULES)?;
    let cls = r.pick(CLASSES)?;
    let err = r.pick(ERRORS)?;

    Ok(format!(
        r#"from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from {mod_}.{imp_mod} import {cls}

logger = logging.getLogger(__name__)


async def {m1}({v1}: {t1}, {v2}: {t2} | None = None) -> {t3}:
    async with _acquire_{v3}({v1}) as {v3}:
        {v2} = await {cls}().{m2}({v3})
        return [{v2} for _ in range(10) if {v2} is not None]


@asynccontextmanager
async def _acquire_{v3}({v1}: {t1}) -> AsyncIterator[{t2}]:
    {v3} = {mod_}.{m3}({v1})
    try:
        yield {v3}
    finally:
        await {v3}.close()


def {m2}_sync({v1}: {t1}, *, max_retries: int = 3) -> {t2}:
    for attempt in range(max_retries):
        try:
            return {mod_}.{m2}({v1})
        except RuntimeError:
            if attempt == max_retries - 1:
                raise
            logger.warning("{err}, attempt %d", attempt + 1)
    raise AssertionError("unreachable")
"#
    ))
}

fn test(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let mod_ = r.pick(MODULES)?;
    let m = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (m[0], m[1], m[2]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let err = r.pick(ERRORS)?;

    Ok(format!(
        r#"import pytest
from unittest.mock import AsyncMock, patch

from {mod_} import {cls}


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
        with patch("{mod_}.{m2}") as mock:
            mock.return_value = {{{{"key": "{v2}"}}}}
            result = await instance.{m1}()
            assert "{v2}" in str(result)
"#
    ))
}

fn http_handler(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let cls_lower = cls.to_lowercase();
    let mod_ = r.pick(MODULES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let route = r.pick(ROUTES)?;
    let table = r.pick(TABLES)?;
    let err = r.pick(ERRORS)?;

    Ok(format!(
        r#"from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from {mod_}.{cls_lower} import {cls}

router = APIRouter(prefix="{route}", tags=["{mod_}"])


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
"#
    ))
}

fn data_model(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let v = r.sample(VARS, 4)?;
    let (v1, v2, v3, v4) = (v[0], v[1], v[2], v[3]);
    let m1 = r.pick(METHODS)?;
    let table = r.pick(TABLES)?;

    Ok(format!(
        r#"from __future__ import annotations

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
"#
    ))
}
