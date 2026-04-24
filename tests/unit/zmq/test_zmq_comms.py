# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for codec-aware ZMQ communication client caching."""

from pathlib import Path

from aiperf.common.channel_codecs import RECORDS_CODEC
from aiperf.common.message_codecs import MsgspecStructCodec, get_message_codec
from aiperf.config import ZMQIPCConfig
from aiperf.zmq.zmq_comms import ZMQIPCCommunication


class TestZMQCommunicationClientCache:
    def test_create_push_client_cache_partitions_by_codec(
        self, mock_zmq_context, tmp_path: Path
    ) -> None:
        """Same address with different codecs should not alias the same cached client."""
        comm = ZMQIPCCommunication(config=ZMQIPCConfig(ipc_path=tmp_path / "ipc"))

        default_client = comm.create_push_client("ipc:///tmp/records.ipc")
        msgpack_client = comm.create_push_client(
            "ipc:///tmp/records.ipc",
            codec=RECORDS_CODEC,
        )
        msgpack_client_again = comm.create_push_client(
            "ipc:///tmp/records.ipc",
            codec=RECORDS_CODEC,
        )

        assert default_client is not msgpack_client
        assert msgpack_client is msgpack_client_again
        assert isinstance(default_client._codec, MsgspecStructCodec)
        assert default_client._codec is get_message_codec()
        assert msgpack_client._codec is RECORDS_CODEC
