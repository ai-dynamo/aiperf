# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class EndpointType(CaseInsensitiveStrEnum):
    """Endpoint types.

    These determine the format of request payload to send to the model.

    Similar to `endpoint_type_map` and `OutputFormat` from `genai-perf`.
    """

    OPENAI_CHAT_COMPLETIONS = "chat"
    OPENAI_COMPLETIONS = "completions"
    # OPENAI_EMBEDDINGS = "embeddings"
    # OPENAI_MULTIMODAL = "multimodal"
    OPENAI_RESPONSES = "responses"

    # TODO: implement other endpoints
    # HUGGINGFACE_GENERATE = "generate"

    # DYNAMIC_GRPC = "dynamic_grpc"
    # NVCLIP = "nvclip"
    # TEMPLATE = "template"

    # RANKINGS = "rankings"
    # IMAGE_RETRIEVAL = "image_retrieval"

    # TENSORRTLLM = "tensorrtllm"
    # TENSORRTLLM_ENGINE = "tensorrtllm_engine"

    # TRITON_GENERATE = "triton_generate"

    # DYNAMO_ENGINE = "dynamo_engine"

    def endpoint_path(self) -> str | None:
        """Get the endpoint path for the endpoint type."""
        endpoint_path_map = {
            # OpenAI endpoints
            EndpointType.OPENAI_CHAT_COMPLETIONS: "/v1/chat/completions",
            # EndpointType.OPENAI_MULTIMODAL: "/v1/chat/completions",
            EndpointType.OPENAI_COMPLETIONS: "/v1/completions",
            # EndpointType.OPENAI_EMBEDDINGS: "/v1/embeddings",
            EndpointType.OPENAI_RESPONSES: "/v1/responses",
            # TODO: implement other endpoints
            # Other
            # EndpointType.NVCLIP: "/v1/embeddings",
            # EndpointType.HUGGINGFACE_GENERATE: "/",  # HuggingFace TGI only exposes root endpoint
            # EndpointType.RANKINGS: "/v1/ranking",  # TODO: Not implemented yet
            # EndpointType.IMAGE_RETRIEVAL: "/v1/infer",  # TODO: Not implemented yet
            # EndpointType.TRITON_GENERATE: "/v2/models/{MODEL_NAME}/generate",  # TODO: Not implemented yet
            # # These endpoints do not have a specific path
            # EndpointType.DYNAMIC_GRPC: None,  # TODO: Not implemented yet
            # EndpointType.TEMPLATE: None,  # TODO: Not implemented yet
            # EndpointType.TENSORRTLLM: None,  # TODO: Not implemented yet
            # EndpointType.TENSORRTLLM_ENGINE: None,  # TODO: Not implemented yet
            # EndpointType.DYNAMO_ENGINE: None,  # TODO: Not implemented yet
        }

        if self not in endpoint_path_map:
            raise NotImplementedError(f"Endpoint not implemented for {self}")

        return endpoint_path_map[self]

    def response_payload_type(self) -> "ResponsePayloadType":
        """Get the response payload type for the request payload type."""
        return ResponsePayloadType.from_endpoint_type(self)


class ResponsePayloadType(CaseInsensitiveStrEnum):
    """Response payload types.

    These determine the format of the response payload that the model will return.

    Equivalent to `output_format` from `genai-perf`.
    """

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_COMPLETIONS = "openai_completions"
    # OPENAI_EMBEDDINGS = "openai_embeddings"
    # OPENAI_MULTIMODAL = "openai_multimodal"
    OPENAI_RESPONSES = "openai_responses"

    # TODO: implement other endpoints
    # HUGGINGFACE_GENERATE = "huggingface_generate"

    # RANKINGS = "rankings"

    # IMAGE_RETRIEVAL = "image_retrieval"

    @classmethod
    def from_endpoint_type(cls, endpoint_type: EndpointType) -> "ResponsePayloadType":
        """Get the response payload type for the endpoint type."""
        endpoint_to_payload_map = {
            EndpointType.OPENAI_CHAT_COMPLETIONS: ResponsePayloadType.OPENAI_CHAT_COMPLETIONS,
            # EndpointType.OPENAI_MULTIMODAL: ResponsePayloadType.OPENAI_CHAT_COMPLETIONS,
            EndpointType.OPENAI_COMPLETIONS: ResponsePayloadType.OPENAI_COMPLETIONS,
            # EndpointType.OPENAI_EMBEDDINGS: ResponsePayloadType.OPENAI_EMBEDDINGS,
            EndpointType.OPENAI_RESPONSES: ResponsePayloadType.OPENAI_RESPONSES,
            # TODO: implement other endpoints
            # EndpointType.HUGGINGFACE_GENERATE: ResponsePayloadType.HUGGINGFACE_GENERATE,
            # EndpointType.RANKINGS: ResponsePayloadType.RANKINGS,
            # EndpointType.IMAGE_RETRIEVAL: ResponsePayloadType.IMAGE_RETRIEVAL,
        }

        if endpoint_type not in endpoint_to_payload_map:
            raise NotImplementedError(
                f"Payload type not implemented for {endpoint_type}"
            )

        return endpoint_to_payload_map[endpoint_type]
