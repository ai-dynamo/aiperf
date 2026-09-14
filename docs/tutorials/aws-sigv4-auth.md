---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: AWS SigV4 Authentication
---

# Benchmarking AWS Endpoints

This guide walks you through benchmarking inference endpoints protected by AWS IAM authentication. AIPerf signs every request with your AWS credentials automatically -- you just need to tell it your AWS region and service name.

`--auth-type sigv4` signs every HTTP request AIPerf sends and works for any SigV4-protected endpoint.

## What's Supported

| Scenario | Non-Streaming | Streaming | Notes |
|----------|:---:|:---:|-------|
| API Gateway + vLLM/TGI/NIM | Yes | Yes | Standard HTTP + SSE |
| SageMaker + vLLM/LMI container | Yes | Yes | Use `--sagemaker-endpoint-name`; see the [SageMaker tutorial](aws-sagemaker.md). AIPerf decodes AWS eventstream framing. |
| Bedrock Converse / InvokeModel | No | No | Different request/response schema -- not OpenAI-compatible |

## Before You Start

1. Install the AWS extra (this pulls in `botocore` for credential handling):

```bash
uv pip install aiperf[aws]
```

2. Make sure your AWS credentials are working:

```bash
aws sts get-caller-identity
```

If that prints your account and role info, you're good to go. If not, see [Setting Up Credentials](#setting-up-credentials).

## Quick Start

The key flags are `--auth-type sigv4`, `--aws-region`, and `--aws-signing-service`. Add these to any `aiperf profile` command and AIPerf will sign every request automatically.

### API Gateway with IAM Auth

Your API Gateway fronts an OpenAI-compatible server and has IAM authorization enabled. Both streaming and non-streaming work:

```bash
aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --streaming \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --request-count 100
```

If your API Gateway maps a custom path to the backend, use `--endpoint` to set it:

```bash
aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com \
    --endpoint /prod/inference/v1/chat/completions \
    --endpoint-type chat \
    --streaming \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --request-count 100
```

### SageMaker with vLLM or LMI (Non-Streaming)

SageMaker endpoints running vLLM or DJL LMI containers accept OpenAI-format request bodies through the `/invocations` path. The response body is passed through unchanged, so non-streaming works. Use `--endpoint` to set the SageMaker invocation path:

```bash
aiperf profile \
    --model my-model \
    --url https://runtime.sagemaker.us-east-1.amazonaws.com \
    --endpoint /endpoints/my-endpoint/invocations \
    --endpoint-type chat \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service sagemaker \
    --request-count 100
```

For SageMaker specifically, prefer `--sagemaker-endpoint-name` over the manual
`--url` + `--endpoint` form shown above -- it derives the URL, the signing scope,
and the correct streaming operation for you, and it supports `--streaming`:

```bash
aiperf profile -m my-model --endpoint-type chat --streaming \
    --sagemaker-endpoint-name my-endpoint --aws-region us-east-1
```

See the [AWS SageMaker tutorial](aws-sagemaker.md) for the full walkthrough. The
manual form above still works for endpoints that need an explicit URL, such as
VPC/PrivateLink.

## Figuring Out Your Region and Service Name

The `--aws-region` should match the region in your endpoint URL:

```text
https://abc123.execute-api.us-east-1.amazonaws.com/...
                           ^^^^^^^^^
                           this is your --aws-region
```

`--aws-signing-service` is the *signing name*: the credential scope the signature
is bound to, which AWS declares separately from the API's own name. The two often
differ, and the flag is named after the signature rather than the service to keep
that distinction visible.

| If your traffic goes through... | Use `--aws-signing-service` |
|--------------------------------|---------------------|
| API Gateway | `execute-api` |
| SageMaker Runtime | `sagemaker` |
| Bedrock Runtime | `bedrock` |

A common gotcha: the signing name isn't always what you'd guess. The
`sagemaker-runtime` API signs as `sagemaker`, and `bedrock-runtime` signs as
`bedrock` -- in both cases the `-runtime` suffix is part of the API id, not the
credential scope. If you get a "SignatureDoesNotMatch" error, this is the first
thing to double-check.

## Setting Up Credentials

If `aws sts get-caller-identity` already works, you can skip this section -- AIPerf will pick up the same credentials automatically.

### Environment Variables (simplest)

Good for quick local testing:

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="wJal..."
export AWS_SESSION_TOKEN="FwoG..."  # only if using temporary credentials

aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --request-count 100
```

### Named Profiles (multiple accounts)

If you work with more than one AWS account, you probably already have profiles set up in `~/.aws/credentials`. Point AIPerf at the right one with `--aws-profile`:

```bash
aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-west-2.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --auth-type sigv4 \
    --aws-region us-west-2 \
    --aws-signing-service execute-api \
    --aws-profile staging \
    --request-count 100
```

Without `--aws-profile`, AIPerf uses whichever credentials the AWS CLI would use by default (environment variables first, then `[default]` profile, then IAM roles).

### SSO

If your team uses AWS IAM Identity Center (SSO), log in first, then pass the profile:

```bash
aws sso login --profile my-sso-profile

aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --aws-profile my-sso-profile \
    --request-count 100
```

### Kubernetes (EKS)

On EKS, credentials are typically injected into your pod automatically via IRSA or Pod Identity. You don't need `--aws-profile` -- just make sure your pod's service account has the right IAM role attached:

```bash
aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --request-count 1000
```

One thing to watch: if your pod has `AWS_ACCESS_KEY_ID` set as an environment variable (e.g., from a Kubernetes Secret), that takes priority over IRSA/Pod Identity. If you're hitting the wrong account, check for stale env vars.

## Long-Running Benchmarks

AIPerf refreshes AWS credentials automatically before each request. This means temporary credentials (from SSO, assumed roles, or IRSA) won't expire mid-benchmark. If you're running a long benchmark with thousands of requests, you don't need to do anything special.

The one exception: if your SSO session itself expires (they typically last 8-12 hours), you'll need to re-run `aws sso login` and restart the benchmark.

## Examples

### High-Throughput API Gateway with Warmup

```bash
aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --streaming \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --request-rate 50 \
    --request-count 1000 \
    --warmup-request-count 20
```

### Multiple API Gateway Endpoints

Distribute load across two endpoints in the same region:

```bash
aiperf profile \
    --model my-model \
    --url https://abc123.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --url https://def456.execute-api.us-east-1.amazonaws.com/prod/v1 \
    --endpoint-type chat \
    --streaming \
    --auth-type sigv4 \
    --aws-region us-east-1 \
    --aws-signing-service execute-api \
    --request-count 500
```

### SageMaker with Custom Dataset

```bash
aiperf profile \
    --model my-model \
    --url https://runtime.sagemaker.us-west-2.amazonaws.com \
    --endpoint /endpoints/my-endpoint/invocations \
    --endpoint-type chat \
    --auth-type sigv4 \
    --aws-region us-west-2 \
    --aws-signing-service sagemaker \
    --dataset prompts.jsonl \
    --dataset-type single_turn
```

## What Signing Does Not Cover

Signing is applied to the inference requests the transport sends. A few features
reach the server over separate code paths that the signer never sees, so their
requests would go out unsigned and be rejected. Rather than let that surface as a
confusing 403 partway into a benchmark, AIPerf rejects these combinations at
startup:

| Combined with `--auth-type` | Why it is rejected |
|---|---|
| `--wait-for-model-timeout` | The readiness probe is an out-of-band request that is not signed. |
| `--reset-kv-cache`, server profiler hooks | Control-plane hooks call the server out of band, unsigned. |
| Multipart endpoints (`image_edit`, `audio_transcription`) | aiohttp streams multipart bodies, so they never materialize as the exact bytes SigV4 must hash. |
| Polling endpoints (`video_generation`) | Job submit/poll runs on a path that bypasses signing. |

Each produces an error naming the specific flag or endpoint type. If you need one
of these against a signed endpoint, drop `--auth-type` or use an unsigned route
to the server.

## Troubleshooting

### "SignatureDoesNotMatch"

This is the most common error. Check these in order:

1. **Is `--aws-region` correct?** It must match the region in the URL.
2. **Is `--aws-signing-service` correct?** See the [service name table](#figuring-out-your-region-and-service-name) above. The names aren't always obvious.
3. **Is your system clock accurate?** AWS rejects signatures that are more than 5 minutes off. Docker containers and VMs are especially prone to clock drift. Run `date -u` and compare to actual UTC.

### "The security token included in the request is expired"

Your temporary credentials have expired. Re-authenticate:

```bash
# For SSO
aws sso login --profile my-profile

# For assumed roles, this usually resolves itself --
# botocore refreshes automatically if the source credentials are still valid
```

### "No AWS credentials found"

AIPerf can't find any credentials. Verify with:

```bash
aws sts get-caller-identity
```

If that also fails, you need to set up credentials -- see [Setting Up Credentials](#setting-up-credentials).

### "SigV4 auth requires botocore"

Install the AWS extra:

```bash
uv pip install aiperf[aws]
```
