# Memory-layout probe image: aiperf-slim + transformers/tokenizers.
#
# The base image already contains the full AIPerf module tree and its deps
# (pydantic, numpy, zmq, orjson, msgspec, aiohttp, uvloop, psutil, rich,
# aiofiles). We only layer on HF transformers + tokenizers, which is the
# piece that isn't in the real forkserver preload today.
#
# Run with /probe.py as the entrypoint; subcommand + flags control the mode.
FROM aiperf-slim:amd64-final

USER 0
ENV VIRTUAL_ENV=/app/.venv
ENV PATH=/app/.venv/bin:$PATH
RUN uv pip install --no-cache-dir \
    transformers==4.46.3 \
    tokenizers \
    sentencepiece \
    protobuf \
    hf_transfer

COPY probe.py /probe.py
COPY tokenizer_preload.py /shared-preload/aiperf_mem_probe_tokenizer_preload.py

# Hugging Face caches live here; the orchestrator can either prefetch
# tokenizers into this path via an initContainer or let the first pod warm
# the cache and rely on page-cache sharing for file-backed tokenizer data.
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /hf_cache /shared && chmod 0777 /hf_cache /shared

ENTRYPOINT ["/app/.venv/bin/python", "-u", "/probe.py"]
