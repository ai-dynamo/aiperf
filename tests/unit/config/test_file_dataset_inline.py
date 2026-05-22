# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from pydantic import ValidationError

from aiperf.common.enums import DatasetFormat, DatasetType
from aiperf.config.dataset import FileDataset


def _base(**overrides):
    return {
        "name": "default",
        "type": DatasetType.FILE,
        "format": DatasetFormat.SINGLE_TURN,
        **overrides,
    }


class TestFileDatasetSourceXOR:
    def test_path_only_validates(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"text": "hi"}\n')
        FileDataset.model_validate(_base(path=str(f)))

    def test_records_only_validates(self):
        FileDataset.model_validate(_base(records=[{"text": "hello"}]))

    def test_both_path_and_records_rejected(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"text": "hi"}\n')
        with pytest.raises(ValidationError) as exc:
            FileDataset.model_validate(_base(path=str(f), records=[{"text": "hi"}]))
        msg = str(exc.value)
        assert "exactly one source" in msg
        assert "path" in msg and "records" in msg

    def test_neither_path_nor_records_rejected(self):
        with pytest.raises(ValidationError) as exc:
            FileDataset.model_validate(_base())
        assert "exactly one source" in str(exc.value)


class TestFileDatasetRecordsShape:
    def test_records_flat_list_for_single_turn(self):
        ds = FileDataset.model_validate(_base(records=[{"text": "hi"}, {"text": "ok"}]))
        assert isinstance(ds.records, list)
        assert len(ds.records) == 2

    def test_records_dict_of_lists_for_random_pool(self):
        ds = FileDataset.model_validate(
            _base(
                format=DatasetFormat.RANDOM_POOL,
                records={"pool_a": [{"text": "a"}], "pool_b": [{"text": "b"}]},
            )
        )
        assert isinstance(ds.records, dict)
        assert set(ds.records.keys()) == {"pool_a", "pool_b"}

    def test_records_dict_rejected_for_non_random_pool(self):
        with pytest.raises(ValidationError) as exc:
            FileDataset.model_validate(_base(records={"pool_a": [{"text": "a"}]}))
        assert "dict-of-lists" in str(exc.value) or "random_pool" in str(exc.value)

    def test_records_empty_list_rejected(self):
        with pytest.raises(ValidationError) as exc:
            FileDataset.model_validate(_base(records=[]))
        assert (
            "empty" in str(exc.value).lower()
            or "at least one" in str(exc.value).lower()
        )

    def test_records_dict_with_empty_pool_rejected(self):
        with pytest.raises(ValidationError) as exc:
            FileDataset.model_validate(
                _base(format=DatasetFormat.RANDOM_POOL, records={"pool_a": []})
            )
        assert (
            "empty" in str(exc.value).lower()
            or "at least one" in str(exc.value).lower()
        )


class TestFileDatasetRecordsSoftWarning:
    def test_warns_above_threshold(self, caplog):
        records = [{"text": f"q{i}"} for i in range(501)]
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.config"):
            FileDataset.model_validate(_base(records=records))
        assert any("inline records" in r.message.lower() for r in caplog.records)

    def test_no_warn_at_or_below_threshold(self, caplog):
        records = [{"text": f"q{i}"} for i in range(500)]
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.config"):
            FileDataset.model_validate(_base(records=records))
        assert not any("inline records" in r.message.lower() for r in caplog.records)

    def test_warns_for_multi_pool_summed(self, caplog):
        records = {
            "pool_a": [{"text": f"a{i}"} for i in range(300)],
            "pool_b": [{"text": f"b{i}"} for i in range(250)],
        }
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.config"):
            FileDataset.model_validate(
                _base(format=DatasetFormat.RANDOM_POOL, records=records)
            )
        assert any("inline records" in r.message.lower() for r in caplog.records)
