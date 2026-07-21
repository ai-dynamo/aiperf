# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models import Image, Turn


def test_copy_with_stripped_media_preserves_cache_only_image_slots() -> None:
    turn = Turn(
        images=[
            Image(
                contents=["data:image/png;base64,AAAA", ""],
                uuids=["uuid-full", "uuid-cached"],
            )
        ]
    )

    copied = turn.copy_with_stripped_media()

    assert copied.images[0].contents == ["image_0", ""]
    assert copied.images[0].uuids == ["uuid-full", "uuid-cached"]
