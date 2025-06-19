#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import SSEFieldType
from aiperf.common.record_models import SSEField, SSEMessage


def parse_sse_message(raw_message: str, perf_ns: int) -> SSEMessage:
    """Parse a raw SSE message into an SSEMessage object.

    Parsing logic based on official HTML SSE Living Standard:
    https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
    """

    message = SSEMessage(perf_ns=perf_ns)
    for line in raw_message.split("\n"):
        if not (line := line.strip()):
            continue

        parts = line.split(":", 1)
        if len(parts) < 2:
            # Fields without a colon have no value, so the whole line is the field name
            message.packets.append(SSEField(name=parts[0].strip(), value=None))
            continue

        field_name, value = parts

        if field_name == "":
            # Field name is empty, so this is a comment
            field_name = SSEFieldType.COMMENT

        message.packets.append(SSEField(name=field_name.strip(), value=value.strip()))

    return message
