#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.services.records_manager.records import Record, Records, Transaction


def test_transaction_dataclass():
    """
    Test the initialization and attribute assignment of the Transaction dataclass.

    This test verifies that a transaction object can be created with a specific timestamp and payload,
    and that its attributes are correctly set and accessible.
    """
    resp = Transaction(timestamp=1234567890, payload={"result": "ok"})
    assert resp.timestamp == 1234567890
    assert resp.payload == {"result": "ok"}


def test_record_dataclass():
    """
    Test the initialization and attribute assignment of the Record dataclass.

    This test creates two transaction objects and a Record object containing them.
    It asserts that the Record's 'request' attribute and 'transactions' list are set correctly.
    """
    resp1 = Transaction(timestamp=1, payload="foo")
    resp2 = Transaction(timestamp=2, payload="bar")
    record = Record(request=100, responses=[resp1, resp2])
    assert record.request == 100
    assert record.responses == [resp1, resp2]


def test_records_add_and_get_records():
    """
    Test the functionality of adding a record to the Records object and retrieving it.

    This test verifies that:
    - A record can be added to the Records instance with a specified request and a list of transaction objects.
    - The added record contains the correct request value and associated transactions.
    """
    records = Records()
    resp1 = Transaction(timestamp=10, payload="payload1")
    resp2 = Transaction(timestamp=20, payload="payload2")
    records.add_record(request=5, responses=[resp1, resp2])
    all_records = records.records
    assert len(all_records) == 1
    assert all_records[0].request == 5
    assert all_records[0].responses == [resp1, resp2]


def test_records_multiple_adds():
    """
    Test that multiple records can be added to the Records object and retrieved in order.

    This test creates two transaction objects with different timestamps and payloads,
    adds them as records with distinct request values, and verifies:
    - The total number of records is correct.
    - The order of records is preserved.
    - The request and transaction payloads are correctly associated with each record.
    """
    records = Records()
    resp1 = Transaction(timestamp=1, payload="a")
    resp2 = Transaction(timestamp=2, payload="b")
    records.add_record(request=100, responses=[resp1])
    records.add_record(request=200, responses=[resp2])
    all_records = records.records
    assert len(all_records) == 2
    assert all_records[0].request == 100
    assert all_records[1].request == 200
    assert all_records[0].responses[0].payload == "a"
    assert all_records[1].responses[0].payload == "b"


def test_records_serialization_and_deserialization():
    """
    Test the serialization and deserialization of the Records model using Pydantic's model_dump and model_validate methods.
    This test creates a Records instance, adds a record with two transaction objects, serializes the Records instance to a dictionary,
    and then deserializes it back to a Records object. It asserts that the deserialized object retains the correct structure and data,
    including the request value and the details of each transaction.
    """

    records = Records()
    req = Transaction(timestamp=1000, payload="request_payload")
    resp1 = Transaction(timestamp=42, payload={"foo": "bar"})
    resp2 = Transaction(timestamp=43, payload=[1, 2, 3])
    records.add_record(request=req, responses=[resp1, resp2])

    # Serialize to dict using Pydantic's model_dump
    data = records.model_dump()

    # Deserialize back using model_validate
    loaded_records = Records.model_validate(data)

    assert len(loaded_records.records) == 1
    assert loaded_records.records[0].request.timestamp == req.timestamp
    assert loaded_records.records[0].request.payload == req.payload
    assert loaded_records.records[0].responses[0].timestamp == 42
    assert loaded_records.records[0].responses[0].payload == {"foo": "bar"}
    assert loaded_records.records[0].responses[1].timestamp == 43
    assert loaded_records.records[0].responses[1].payload == [1, 2, 3]
