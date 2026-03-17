# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from unittest.mock import MagicMock, patch

import numpy as np

from aiperf.common.models.modality_token_counts import ModalityTokenCounts
from aiperf.common.processor import Processor, _decode_data_uri


class TestDecodeDataUri:
    def test_decodes_base64_data_uri(self):
        from PIL import Image as PILImage

        # Create a minimal valid PNG in memory
        img = PILImage.new("RGB", (1, 1), color="red")
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        result = _decode_data_uri(data_uri)
        assert isinstance(result, PILImage.Image)

    def test_decodes_raw_base64(self):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (2, 2), color="blue")
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = _decode_data_uri(b64)
        assert isinstance(result, PILImage.Image)


class TestProcessor:
    def test_from_pretrained_returns_processor_on_success(self):
        mock_auto = MagicMock()
        with patch(
            "transformers.AutoProcessor.from_pretrained", return_value=mock_auto
        ):
            proc = Processor.from_pretrained("test-model")
            assert proc is not None
            assert isinstance(proc, Processor)

    def test_returns_none_when_no_processor_available(self):
        with patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=Exception("not found"),
        ):
            proc = Processor.from_pretrained("nonexistent-model")
            assert proc is None

    def test_estimate_input_modality_counts(self):
        mock_proc = MagicMock()

        proc = Processor(mock_proc)

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"input_ids": np.zeros((1, 600))}  # total
            return {"input_ids": np.zeros((1, 120))}  # text-only

        mock_proc.side_effect = side_effect
        mock_proc.return_value = None

        # Use raw base64 PNG for testing
        import io

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (1, 1))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = proc.estimate_input_modality_counts(
            ["hello"], [f"data:image/png;base64,{b64}"]
        )
        assert isinstance(result, ModalityTokenCounts)
        assert result.text == 120
        assert result.image == 480
        assert result.total == 600
