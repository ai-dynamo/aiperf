// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Seeded structural renderers for the native coding corpus.

use aiperf_rng::RandomGenerator;
use serde_json::json;

use super::vocab::*;
use crate::recorded::RecordedTraceError;

#[derive(Clone, Copy, Debug)]
pub(super) enum TemplateKind {
    Python,
    Go,
    Rust,
    TypeScript,
    MlTraining,
    MlInference,
    MlConfig,
    BashOutput,
    MlTrainingLog,
    JsonResponse,
    ErrorTraceback,
    CudaError,
    Sql,
    UserPrompt,
    ToolUse,
    Conversation,
    GitDiff,
    Cicd,
    Config,
    Markdown,
    TestOutput,
}

pub(super) struct TemplateRenderer {
    random: RandomGenerator,
}

impl TemplateRenderer {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            random: RandomGenerator::from_seed(Some(seed)),
        }
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        self.random.shuffle(values);
    }

    pub(super) fn render(
        &mut self,
        kind: TemplateKind,
        ordinal: usize,
    ) -> Result<String, RecordedTraceError> {
        match kind {
            TemplateKind::Python => self.python(),
            TemplateKind::Go => self.go(),
            TemplateKind::Rust => self.rust(),
            TemplateKind::TypeScript => self.typescript(),
            TemplateKind::MlTraining => self.ml_training(ordinal),
            TemplateKind::MlInference => self.ml_inference(),
            TemplateKind::MlConfig => self.ml_config(),
            TemplateKind::BashOutput => self.bash_output(),
            TemplateKind::MlTrainingLog => self.ml_training_log(ordinal),
            TemplateKind::JsonResponse => self.json_response(ordinal),
            TemplateKind::ErrorTraceback => self.error_traceback(),
            TemplateKind::CudaError => self.cuda_error(),
            TemplateKind::Sql => self.sql(),
            TemplateKind::UserPrompt => self.user_prompt(),
            TemplateKind::ToolUse => self.tool_use(),
            TemplateKind::Conversation => self.conversation(),
            TemplateKind::GitDiff => self.git_diff(),
            TemplateKind::Cicd => self.cicd(ordinal),
            TemplateKind::Config => self.config(),
            TemplateKind::Markdown => self.markdown(),
            TemplateKind::TestOutput => self.test_output(),
        }
    }

    fn pick(
        &mut self,
        values: &'static [&'static str],
    ) -> Result<&'static str, RecordedTraceError> {
        self.random
            .choice(values)
            .copied()
            .map_err(|error| RecordedTraceError(error.to_string()))
    }

    fn number(&mut self, low: i64, high: i64) -> Result<i64, RecordedTraceError> {
        self.random
            .randint(low, high)
            .map_err(|error| RecordedTraceError(error.to_string()))
    }

    fn python(&mut self) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let class = self.pick(CLASSES)?;
        let method = self.pick(METHODS)?;
        let helper = self.pick(METHODS)?;
        let var = self.pick(VARS)?;
        let ty = self.pick(TYPES)?;
        let error = self.pick(ERRORS)?;
        Ok(format!(
            r#"from __future__ import annotations

import asyncio
import logging
from {module}.service import {class}

logger = logging.getLogger(__name__)


class {class}Controller:
    \"\"\"Coordinates {method} operations for {module}.\"\"\"

    def __init__(self, service: {class}, max_retries: int = 3) -> None:
        self._service = service
        self._max_retries = max_retries

    async def {method}(self, {var}: {ty}) -> dict[str, object]:
        for attempt in range(self._max_retries):
            try:
                result = await self._service.{helper}({var})
                return {{\"status\": \"ok\", \"result\": result}}
            except RuntimeError as exc:
                logger.warning(\"{error}: attempt %d\", attempt + 1)
                if attempt + 1 == self._max_retries:
                    raise
                await asyncio.sleep(2 ** attempt / 10)
        raise AssertionError(\"unreachable\")
"#
        ))
    }

    fn go(&mut self) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let class = self.pick(CLASSES)?;
        let method = self.pick(METHODS)?;
        let var = self.pick(VARS)?;
        let route = self.pick(ROUTES)?;
        Ok(format!(
            r#"package {module}

import (
    \"context\"
    \"encoding/json\"
    \"fmt\"
    \"net/http\"
    \"time\"
)

type {class} struct {{
    client *http.Client
    retries int
}}

func (s *{class}) {method}(ctx context.Context, {var} string) (map[string]any, error) {{
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()
    req, err := http.NewRequestWithContext(ctx, http.MethodPost, \"{route}\", nil)
    if err != nil {{ return nil, fmt.Errorf(\"build request: %w\", err) }}
    resp, err := s.client.Do(req)
    if err != nil {{ return nil, fmt.Errorf(\"dispatch {var}: %w\", err) }}
    defer resp.Body.Close()
    var out map[string]any
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {{
        return nil, fmt.Errorf(\"decode response: %w\", err)
    }}
    return out, nil
}}
"#
        ))
    }

    fn rust(&mut self) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let class = self.pick(CLASSES)?;
        let method = self.pick(METHODS)?;
        let var = self.pick(VARS)?;
        let error = self.pick(ERRORS)?;
        Ok(format!(
            r#"use std::sync::Arc;

use anyhow::Context;
use serde::{{Deserialize, Serialize}};
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {class}Request {{
    pub {var}: String,
    pub timeout_ms: u64,
}}

pub struct {class}<C> {{
    client: Arc<C>,
    permits: Arc<Semaphore>,
}}

impl<C: {module}::Client> {class}<C> {{
    pub async fn {method}(&self, request: {class}Request) -> anyhow::Result<String> {{
        let _permit = self.permits.acquire().await.context(\"{error}\")?;
        self.client
            .execute(request.{var})
            .await
            .with_context(|| format!(\"{method} failed after {{}} ms\", request.timeout_ms))
    }}
}}
"#
        ))
    }

    fn typescript(&mut self) -> Result<String, RecordedTraceError> {
        let class = self.pick(CLASSES)?;
        let method = self.pick(METHODS)?;
        let var = self.pick(VARS)?;
        let route = self.pick(ROUTES)?;
        Ok(format!(
            r#"import {{ z }} from 'zod';

const {class}Schema = z.object({{
  {var}: z.string().min(1),
  timeoutMs: z.number().int().positive().default(30_000),
}});

export type {class}Request = z.infer<typeof {class}Schema>;

export class {class}Client {{
  constructor(private readonly baseUrl: string) {{}}

  async {method}(input: {class}Request, signal?: AbortSignal): Promise<unknown> {{
    const payload = {class}Schema.parse(input);
    const response = await fetch(`${{this.baseUrl}}{route}`, {{
      method: 'POST',
      headers: {{ 'content-type': 'application/json' }},
      body: JSON.stringify(payload),
      signal,
    }});
    if (!response.ok) throw new Error(`request failed: ${{response.status}}`);
    return response.json();
  }}
}}
"#
        ))
    }

    fn ml_training(&mut self, ordinal: usize) -> Result<String, RecordedTraceError> {
        let model = self.pick(MODELS)?;
        let batch = self.number(1, 64)?;
        let steps = self.number(100, 20_000)?;
        Ok(format!(
            r#"import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = \"{model}\"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.gradient_checkpointing_enable()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)

for step, batch in enumerate(DataLoader(dataset, batch_size={batch}, shuffle=True)):
    with torch.autocast(\"cuda\", dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss / 8
    loss.backward()
    if (step + 1) % 8 == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if step >= {steps}:
        break

# deterministic corpus block {ordinal}
"#
        ))
    }

    fn ml_inference(&mut self) -> Result<String, RecordedTraceError> {
        let model = self.pick(MODELS)?;
        let max_tokens = self.number(64, 4096)?;
        Ok(format!(
            r#"from vllm import LLM, SamplingParams

llm = LLM(
    model=\"{model}\",
    tensor_parallel_size=8,
    max_model_len=32768,
    gpu_memory_utilization=0.92,
)
sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens={max_tokens})
prompts = [row[\"prompt\"] for row in dataset]
for output in llm.generate(prompts, sampling, use_tqdm=True):
    request_id = output.request_id
    text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    print({{\"request_id\": request_id, \"tokens\": len(token_ids), \"text\": text}})
"#
        ))
    }

    fn ml_config(&mut self) -> Result<String, RecordedTraceError> {
        let model = self.pick(MODELS)?;
        let batch = self.number(1, 64)?;
        Ok(format!(
            r#"model:
  name_or_path: {model}
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
training:
  per_device_train_batch_size: {batch}
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  bf16: true
  gradient_checkpointing: true
distributed:
  backend: nccl
  zero_stage: 3
  tensor_parallel_size: 8
logging:
  report_to: [wandb, tensorboard]
  logging_steps: 10
"#
        ))
    }

    fn bash_output(&mut self) -> Result<String, RecordedTraceError> {
        let command = self.pick(COMMANDS)?;
        let module = self.pick(MODULES)?;
        let elapsed = self.number(1, 9999)?;
        Ok(format!(
            r#"$ {command}
[12:04:31] loading configuration for {module}
[12:04:31] resolving dependencies
[12:04:32] compiling 47 targets
[12:04:34] running validation checks
{module}::health ........ ok
{module}::integration ... ok
{module}::load .......... ok

completed successfully in {elapsed} ms
peak memory: 842 MiB; cache hits: 1,284; cache misses: 17
"#
        ))
    }

    fn ml_training_log(&mut self, ordinal: usize) -> Result<String, RecordedTraceError> {
        let loss = self.number(120, 890)? as f64 / 100.0;
        let tokens = self.number(12_000, 240_000)?;
        let throughput = self.number(800, 24_000)?;
        Ok(format!(
            "2026-03-14 09:42:{:02} | rank=0 | epoch={:.2} | step={} | loss={loss:.4} | lr=1.82e-5\n\
             2026-03-14 09:42:{:02} | tokens={} | tokens/s={} | grad_norm=0.842 | scale=65536\n\
             2026-03-14 09:42:{:02} | gpu_mem=71.4GiB | reserved=76.2GiB | data_time=0.018s | step_time=1.284s",
            ordinal % 60,
            ordinal as f64 / 20.0,
            ordinal * 100,
            (ordinal + 1) % 60,
            tokens,
            throughput,
            (ordinal + 2) % 60,
        ))
    }

    fn json_response(&mut self, ordinal: usize) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let status = self.number(200, 299)?;
        let count = self.number(1, 5000)?;
        serde_json::to_string_pretty(&json!({
            "id": format!("req-{ordinal:08x}"),
            "object": "list",
            "created": 1_774_000_000_u64 + ordinal as u64,
            "status": status,
            "service": module,
            "data": [
                {"index": 0, "state": "ready", "tokens": count},
                {"index": 1, "state": "running", "tokens": count + 17},
            ],
            "usage": {"prompt_tokens": count, "completion_tokens": 128, "total_tokens": count + 128},
        }))
        .map_err(|error| RecordedTraceError(error.to_string()))
    }

    fn error_traceback(&mut self) -> Result<String, RecordedTraceError> {
        let file = self.pick(FILES)?;
        let module = self.pick(MODULES)?;
        let method = self.pick(METHODS)?;
        let error = self.pick(ERRORS)?;
        Ok(format!(
            r#"Traceback (most recent call last):
  File \"{file}\", line 184, in <module>
    asyncio.run(main())
  File \"/usr/lib/python3.12/asyncio/runners.py\", line 194, in run
    return runner.run(main)
  File \"{file}\", line 129, in {method}
    await {module}.dispatch(request)
  File \"src/{module}/service.py\", line 87, in dispatch
    raise RuntimeError(\"{error}\")
RuntimeError: {error}

During handling of the above exception, another exception occurred:
ConnectionError: request failed after 3 retries
"#
        ))
    }

    fn cuda_error(&mut self) -> Result<String, RecordedTraceError> {
        let error = self.pick(CUDA_ERRORS)?;
        let model = self.pick(MODELS)?;
        Ok(format!(
            r#"[rank0]: Traceback (most recent call last):
[rank0]:   File \"train.py\", line 412, in training_step
[rank0]:     loss = model(input_ids=input_ids, attention_mask=attention_mask).loss
[rank0]:   File \"torch/nn/modules/module.py\", line 1751, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]: RuntimeError: {error}

model={model} rank=0 local_rank=0 world_size=8
allocated=77.31GiB reserved=78.02GiB free=384MiB
Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.
"#
        ))
    }

    fn sql(&mut self) -> Result<String, RecordedTraceError> {
        let table = self.pick(TABLES)?;
        let other = self.pick(TABLES)?;
        let var = self.pick(VARS)?;
        Ok(format!(
            r#"WITH recent AS (
    SELECT id, tenant_id, created_at, status,
           row_number() OVER (PARTITION BY tenant_id ORDER BY created_at DESC) AS rank
    FROM {table}
    WHERE created_at >= now() - interval '30 days'
      AND status = $1
), aggregated AS (
    SELECT tenant_id, count(*) AS {var}, max(created_at) AS last_seen
    FROM recent WHERE rank <= 100 GROUP BY tenant_id
)
SELECT a.tenant_id, a.{var}, a.last_seen, o.display_name
FROM aggregated a
LEFT JOIN {other} o ON o.tenant_id = a.tenant_id
WHERE a.{var} > 10
ORDER BY a.{var} DESC
LIMIT 100;
"#
        ))
    }

    fn user_prompt(&mut self) -> Result<String, RecordedTraceError> {
        let verb = self.pick(PROMPT_VERBS)?;
        let module = self.pick(MODULES)?;
        let method = self.pick(METHODS)?;
        let file = self.pick(FILES)?;
        let error = self.pick(ERRORS)?;
        Ok(format!(
            "Please {verb} the `{module}` implementation in `{file}`. The `{method}` path currently fails with \"{error}\" under concurrent load. Preserve the public API, add focused unit and integration tests, and explain any compatibility tradeoffs."
        ))
    }

    fn tool_use(&mut self) -> Result<String, RecordedTraceError> {
        let command = self.pick(COMMANDS)?;
        let file = self.pick(FILES)?;
        let method = self.pick(METHODS)?;
        Ok(format!(
            r#"assistant to=functions.exec_command
{{"cmd":"sed -n '1,240p' {file} && {command}","yield_time_ms":1000}}
tool
Chunk ID: 42da91
Process exited with code 0
Final output:
Found the `{method}` implementation and its three call sites. Two tests exercise the happy path; the timeout branch has no regression coverage.
"#
        ))
    }

    fn conversation(&mut self) -> Result<String, RecordedTraceError> {
        let verb = self.pick(PROMPT_VERBS)?;
        let module = self.pick(MODULES)?;
        let method = self.pick(METHODS)?;
        let var = self.pick(VARS)?;
        let error = self.pick(ERRORS)?;
        Ok(format!(
            r#"user
Can you {verb} the {module} service? `{method}` returns {error} whenever {var} is empty.

assistant
I’ll trace the request validation and the downstream call before changing behavior.

assistant to=functions.exec_command
{{"cmd":"rg -n \"{method}|{var}\" src tests"}}

tool
src/{module}/service.py:84:async def {method}({var}):
tests/test_{module}.py:41:async def test_{method}_success():

assistant
The validator accepts an empty value, but the repository assumes a non-empty key. I’ll make the boundary explicit and add a regression test for both `None` and the empty string.

user
Please keep the old callers working and include the integration test.
"#
        ))
    }

    fn git_diff(&mut self) -> Result<String, RecordedTraceError> {
        let file = self.pick(FILES)?;
        let method = self.pick(METHODS)?;
        let var = self.pick(VARS)?;
        Ok(format!(
            r#"diff --git a/{file} b/{file}
index 8d731ab..ca91f07 100644
--- a/{file}
+++ b/{file}
@@ -42,8 +42,15 @@ async def {method}({var}):
-    return await client.execute({var})
+    if {var} is None:
+        raise ValueError(\"{var} must be provided\")
+    try:
+        return await client.execute({var})
+    except TimeoutError as exc:
+        metrics.increment(\"{method}.timeout\")
+        raise ServiceUnavailable(\"dependency timed out\") from exc
 
 async def close():
     await client.close()
"#
        ))
    }

    fn cicd(&mut self, ordinal: usize) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        Ok(format!(
            r#"Run make check
  shell: /usr/bin/bash -e {{0}}
  env:
    CI: true
    RUST_BACKTRACE: 1

Checking formatting...................................................passed
Checking SPDX headers................................................passed
Running {module} unit tests..........................................passed
Running integration shard {}/8.......................................passed
Building release artifacts...........................................passed

test result: ok. 184 passed; 0 failed; 3 ignored; finished in 18.42s
Uploading coverage to artifact store
Coverage: 91.7% lines, 88.4% branches
"#,
            ordinal % 8 + 1
        ))
    }

    fn config(&mut self) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let route = self.pick(ROUTES)?;
        Ok(format!(
            r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {module}
  labels:
    app.kubernetes.io/name: {module}
spec:
  replicas: 4
  selector:
    matchLabels: {{ app: {module} }}
  template:
    metadata:
      labels: {{ app: {module} }}
    spec:
      containers:
        - name: api
          image: registry.example.com/{module}:2026.03.14
          ports: [{{ containerPort: 8080 }}]
          readinessProbe:
            httpGet: {{ path: {route}, port: 8080 }}
          resources:
            requests: {{ cpu: \"2\", memory: 4Gi }}
            limits: {{ cpu: \"8\", memory: 16Gi }}
"#
        ))
    }

    fn markdown(&mut self) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let method = self.pick(METHODS)?;
        Ok(format!(
            r#"# {module} service

The `{module}` service owns request validation, bounded retries, and lifecycle metrics.

## Usage

```python
client = {module}.Client(endpoint=\"http://localhost:8080\")
result = await client.{method}(payload, timeout=30.0)
```

## Operational guarantees

- Requests carry a stable correlation ID.
- Retryable errors use exponential backoff with jitter.
- Shutdown drains admitted work before closing the connection pool.
- Metrics distinguish validation, transport, timeout, and server failures.

## Development

Run `make check` before opening a pull request. Integration tests require Docker and reserve an ephemeral local port.
"#
        ))
    }

    fn test_output(&mut self) -> Result<String, RecordedTraceError> {
        let module = self.pick(MODULES)?;
        let method = self.pick(METHODS)?;
        Ok(format!(
            r#"============================= test session starts ==============================
platform linux -- Python 3.12.9, pytest-8.3.5, pluggy-1.5.0
rootdir: /workspace/{module}
plugins: asyncio-0.25.3, timeout-2.3.1, xdist-3.6.1
collected 184 items

tests/unit/test_{module}.py ........................................... [ 24%]
tests/unit/test_{method}.py .......................................... [ 48%]
tests/integration/test_api.py ........................................ [ 72%]
tests/integration/test_database.py ................................... [ 96%]
tests/e2e/test_smoke.py .......                                        [100%]

======================= 184 passed, 3 warnings in 18.42s =======================
"#
        ))
    }
}
