// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const SENSITIVE_CONFIG_KEYS = ['api_key', 'apiKey', 'authorization', 'bearerToken', 'client_secret', 'password', 'secret', 'secretRef', 'token'];

function isSensitiveConfigKey(key) {
  const normalized = String(key).toLowerCase().replace(/[^a-z0-9]/g, '');
  return SENSITIVE_CONFIG_KEYS.some((example) => {
    const needle = example.toLowerCase().replace(/[^a-z0-9]/g, '');
    return normalized === needle || normalized.includes(needle);
  });
}

export function redactConfigForYaml(value) {
  if (Array.isArray(value)) return value.map((item) => redactConfigForYaml(item));
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    isSensitiveConfigKey(key) ? '[REDACTED]' : redactConfigForYaml(item),
  ]));
}
