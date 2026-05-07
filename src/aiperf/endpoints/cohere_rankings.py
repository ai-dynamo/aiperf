# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint

CohereDocument = str | dict[str, list[dict[str, Any]]]


class CohereRankingsEndpoint(BaseRankingsEndpoint):
    """Cohere Rankings Endpoint."""

    def build_payload(
        self,
        query_text: str,
        passages: Sequence[str],
        model_name: str,
        *,
        images: Sequence[str] = (),
        videos: Sequence[str] = (),
        audios: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Build payload to match Cohere Rankings API schema."""
        if audios:
            raise ValueError("Cohere rankings does not support audio input.")

        payload = {
            "model": model_name,
            "query": query_text,
            "documents": self._build_documents(
                passages=passages,
                images=images,
                videos=videos,
            ),
        }
        return payload

    def _build_documents(
        self,
        *,
        passages: Sequence[str],
        images: Sequence[str],
        videos: Sequence[str],
    ) -> list[CohereDocument]:
        """Build Cohere/vLLM rerank documents from index-paired modalities."""
        self._validate_document_counts(
            passages=passages,
            images=images,
            videos=videos,
        )

        if not images and not videos:
            return list(passages)

        document_count = self._document_count(
            passages=passages,
            images=images,
            videos=videos,
        )
        documents: list[CohereDocument] = []
        for index in range(document_count):
            content: list[dict[str, Any]] = []
            if passages:
                content.append({"type": "text", "text": passages[index]})
            if images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": images[index]},
                    }
                )
            if videos:
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": videos[index]},
                    }
                )
            documents.append({"content": content})
        return documents

    def _validate_document_counts(
        self,
        *,
        passages: Sequence[str],
        images: Sequence[str],
        videos: Sequence[str],
    ) -> None:
        """Ensure non-empty document modalities can be paired by index."""
        counts = {
            "passages": len(passages),
            "images": len(images),
            "videos": len(videos),
        }
        non_zero_counts = {name: count for name, count in counts.items() if count}
        if not non_zero_counts:
            return

        expected_count = next(iter(non_zero_counts.values()))
        mismatches = {
            name: count
            for name, count in non_zero_counts.items()
            if count != expected_count
        }
        if mismatches:
            counts_str = ", ".join(
                f"{name}={count}" for name, count in non_zero_counts.items()
            )
            raise ValueError(
                "Cohere rankings multimodal documents must be index-paired "
                f"with matching non-zero counts. Got {counts_str}."
            )

    def _document_count(
        self,
        *,
        passages: Sequence[str],
        images: Sequence[str],
        videos: Sequence[str],
    ) -> int:
        """Return the number of multimodal documents after count validation."""
        for contents in (passages, images, videos):
            if contents:
                return len(contents)
        return 0

    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ranking results from Cohere Rankings API response."""
        results = json_obj.get("results", [])
        rankings = [
            {"index": r.get("index"), "score": r.get("relevance_score")}
            for r in results
        ]
        return rankings
