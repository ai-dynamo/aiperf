"""Effective-``store`` resolution for FORK-child chaining decisions.

``Worker._turn_store_persisted`` decides whether a single parent turn was
server-persisted, which gates whether a WebSocket FORK child can chain onto an
inherited ``previous_response_id`` or must replay its history. ``raw_payload``
turns bypass ``ResponsesEndpoint.format_payload`` entirely, so their own
``store`` key -- not the endpoint-wide ``extra`` or per-turn ``extra_body`` -- is
authoritative.
"""

import pytest
from pytest import param

from aiperf.common.models.dataset_models import Turn
from aiperf.workers.worker import Worker


@pytest.mark.parametrize(
    "raw_payload,extra_body,endpoint_store,expected",
    [
        param({"store": True}, None, False, True, id="raw-store-true-wins"),
        param({"store": False}, None, True, False, id="raw-store-false-wins"),
        param({}, None, True, False, id="raw-no-store-not-persisted"),
        param(None, {"store": True}, False, True, id="extra-body-override-true"),
        param(None, {"store": False}, True, False, id="extra-body-override-false"),
        param(None, None, True, True, id="endpoint-default-true"),
        param(None, None, False, False, id="endpoint-default-false"),
    ],
)  # fmt: skip
def test_turn_store_persisted(
    raw_payload: dict | None,
    extra_body: dict | None,
    endpoint_store: bool,
    expected: bool,
) -> None:
    turn = Turn(raw_payload=raw_payload, extra_body=extra_body)
    assert Worker._turn_store_persisted(turn, endpoint_store) is expected
