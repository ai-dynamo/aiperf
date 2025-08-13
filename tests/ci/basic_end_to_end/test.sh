#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: switch to the commented code when dynamo 0.4.1 is released
# timeout 5m /bin/bash -c 'while ! curl -s localhost:8080/v1/models | jq -en "input | (.data // []) | length > 0" > /dev/null 2>&1; do sleep 1; done'
timeout 5m /bin/bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '"'"'{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"a"}],"max_completion_tokens":1}'"'"')" != 200 ]; do sleep 1; done'

if [ $? -eq 124 ]; then
  echo -e "\033[0;36m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;36m║           *** SERVER LOG ***           ║\033[0m"
  echo -e "\033[0;36m╚════════════════════════════════════════╝\033[0m"
  cat server.log
  echo -e "\033[0;31m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;31m║         *** TIMEOUT ERROR ***          ║\033[0m"
  echo -e "\033[0;31m║ Server did not start within 5 minutes  ║\033[0m"
  echo -e "\033[0;31m║          See server log above          ║\033[0m"
  echo -e "\033[0;31m╚════════════════════════════════════════╝\033[0m"
  exit 1
fi

echo -e "\033[0;36m╔════════════════════════════════════════╗\033[0m"
echo -e "\033[0;36m║           *** AIPERF LOG ***           ║\033[0m"
echo -e "\033[0;36m╚════════════════════════════════════════╝\033[0m"

timeout 5m aiperf profile --model-names ${MODEL} -u localhost:8080 --concurrency 100 --request-count 300

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 124 ]; then
  echo -e "\033[0;36m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;36m║           *** SERVER LOG ***           ║\033[0m"
  echo -e "\033[0;36m╚════════════════════════════════════════╝\033[0m"
  cat server.log
  echo -e "\033[0;31m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;31m║         *** TIMEOUT ERROR ***          ║\033[0m"
  echo -e "\033[0;31m║ aiperf did not finish within 5 minutes ║\033[0m"
  echo -e "\033[0;31m║    See aiperf and server logs above    ║\033[0m"
  echo -e "\033[0;31m╚════════════════════════════════════════╝\033[0m"
  exit 1
elif [ ${EXIT_CODE} -ne 0 ]; then
  echo -e "\033[0;36m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;36m║           *** SERVER LOG ***           ║\033[0m"
  echo -e "\033[0;36m╚════════════════════════════════════════╝\033[0m"
  cat server.log
  echo -e "\033[0;31m╔════════════════════════════════════════╗\033[0m"
  echo -e "\033[0;31m║    *** NONZERO EXIT CODE ERROR ***     ║\033[0m"
  echo -e "\033[0;31m║   aiperf returned nonzero exit code    ║\033[0m"
  echo -e "\033[0;31m║    See aiperf and server logs above    ║\033[0m"
  echo -e "\033[0;31m╚════════════════════════════════════════╝\033[0m"
  exit 1
fi
