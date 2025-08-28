#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

time docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

DYNAMO_REPO_TAG=$(docker run --rm --entrypoint "" ${DYNAMO_PREBUILT_IMAGE_TAG} cat /workspace/version.txt | cut -d'+' -f2)

curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml

docker compose -f docker-compose.yml down || true
docker compose -f docker-compose.yml up -d

docker run \
  --rm \
  --gpus all \
  --network host \
  "${DYNAMO_PREBUILT_IMAGE_TAG}" \
  /bin/bash -c "python3 -m dynamo.frontend --http-port ${PORT}& \
                python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching" \
  >> server.log 2>&1 &

sleep 2

timeout 5m bash -c "until curl -fsS http://localhost:${PORT}/v1/models \
  | jq -en 'input | (.data // []) | length > 0' >/dev/null 2>&1; do sleep 1; done"
rc=$?
if [ $rc -eq 124 ]; then
  echo -e "\033[0;36m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;36m║           *** SERVER LOG ***           ║\033[0m"
  echo -e "\033[0;36m╚════════════════════════════════════════╝\033[0m"
  tail -n 200 server.log || true
  echo -e "\033[0;31m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;31m║         *** TIMEOUT ERROR ***          ║\033[0m"
  echo -e "\033[0;31m║ Server did not start within 5 minutes  ║\033[0m"
  echo -e "\033[0;31m║          See server log above          ║\033[0m"
  echo -e "\033[0;31m╚════════════════════════════════════════╝\033[0m"
  exit 1
fi
