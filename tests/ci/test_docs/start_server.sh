#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

export DYNAMO_REPO_TAG=$(docker run --rm --entrypoint "" ${DYNAMO_PREBUILT_IMAGE_TAG} cat /workspace/version.txt | cut -d'+' -f2)

# TODO: switch to this when dynamo 0.4.0 is released
curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml
#curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/metrics/docker-compose.yml

docker compose -f docker-compose.yml down || true

docker compose -f docker-compose.yml up -d

time docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/container/run.sh

chmod +x run.sh

# TODO: switch to this when dynamo 0.4.0 is released
./run.sh --image ${DYNAMO_PREBUILT_IMAGE_TAG} -- /bin/bash -c "python3 -m dynamo.frontend & python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching" > server.log 2>&1 &
#./run.sh --image ${DYNAMO_PREBUILT_IMAGE_TAG} -- /bin/bash -c "dynamo run in=http out=vllm ${MODEL}" > server.log 2>&1 &

timeout 5m /bin/bash -c 'while ! curl -s localhost:8080/v1/models | jq -en "input | (.data // []) | length > 0" > /dev/null 2>&1; do sleep 1; done'

if [ $? -eq 124 ]; then
  cat server.log
  echo -e "\033[0;31m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;31m║         *** TIMEOUT ERROR ***          ║\033[0m"
  echo -e "\033[0;31m║  Server did not start after 5 minutes  ║\033[0m"
  echo -e "\033[0;31m║          See server log above          ║\033[0m"
  echo -e "\033[0;31m╚════════════════════════════════════════╝\033[0m"
  exit 1
fi
