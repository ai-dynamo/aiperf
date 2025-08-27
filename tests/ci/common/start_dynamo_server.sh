#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

time docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml

docker compose -f docker-compose.yml down || true

docker compose -f docker-compose.yml up -d

time docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/container/run.sh

chmod +x run.sh

./run.sh --image ${DYNAMO_PREBUILT_IMAGE_TAG} -- /bin/bash -c "python3 -m dynamo.frontend --port 8000& python3 -m dynamo.vllm --model ${MODEL} --port 8000 --enforce-eager --no-enable-prefix-caching" > server.log 2>&1 &

timeout 5m /bin/bash -c 'while ! curl -s localhost:8000/v1/models | jq -en "input | (.data // []) | length > 0" > /dev/null 2>&1; do sleep 1; done'

if [ $? -eq 124 ]; then
  cat server.log
  echo -e "\033[0;31m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;31m║         *** TIMEOUT ERROR ***          ║\033[0m"
  echo -e "\033[0;31m║  Server did not start after 5 minutes  ║\033[0m"
  echo -e "\033[0;31m║          See server log above          ║\033[0m"
  echo -e "\033[0;31m╚════════════════════════════════════════╝\033[0m"
  exit 1
fi
