#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

time docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

DYNAMO_REPO_TAG=$(docker run --rm --entrypoint "" ${DYNAMO_PREBUILT_IMAGE_TAG} cat /workspace/version.txt | cut -d'+' -f2)

curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml

docker compose -f docker-compose.yml down || true

docker compose -f docker-compose.yml up -d

curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/container/run.sh

chmod +x run.sh

./run.sh --image ${DYNAMO_PREBUILT_IMAGE_TAG} -- /bin/bash -c "python3 -m dynamo.frontend & python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching" > ${AIPERF_SOURCE_DIR}/server.log 2>&1 &
