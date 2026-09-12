# Benchmarking an AWS SageMaker Endpoint

AIPerf can drive a model hosted behind a SageMaker Runtime endpoint directly --
no translation shim in front of it.

SageMaker is not an OpenAI-compatible HTTP server. Requests go to
`POST /endpoints/{name}/invocations`, must be SigV4-signed, and streaming
responses come back as binary `application/vnd.amazon.eventstream` frames rather
than SSE. AIPerf handles all three.

What it does *not* change is the request body. If you are running vLLM, TGI, or
an LMI container behind the endpoint, that container still speaks OpenAI, so
`--endpoint-type chat` (or `completions`, or `embeddings`) works exactly as it
does against a local server.

## Quick Start

```bash
aiperf profile \
    -m my-model \
    --endpoint-type chat \
    --streaming \
    --sagemaker-endpoint-name my-vllm-endpoint \
    --aws-region us-west-2
```

That is the whole thing. `--sagemaker-endpoint-name` implies the rest:

| Derived | From |
|---|---|
| Transport (`sagemaker`) | the presence of `--sagemaker-endpoint-name` |
| SigV4 request signing | the SageMaker transport |
| Signing scope (`sagemaker`) | botocore's own `sagemaker-runtime` service model |
| Base URL (`https://runtime.sagemaker.us-west-2.amazonaws.com`) | `--aws-region` |
| Request path (`/endpoints/my-vllm-endpoint/invocations…`) | the endpoint name |
| `invocations` vs `invocations-response-stream` | `--streaming` |

Install the AWS extra first if you have not already:

```bash
uv pip install 'aiperf[aws]'
```

## The One Surprising Part

**`--endpoint-type chat` still applies, even though `/v1/chat/completions` never
appears in the URL.**

This trips up nearly everyone the first time. SageMaker routes on the request
*body*, not the path: every endpoint type is invoked through the same
`/endpoints/{name}/invocations` path, and the container behind it decides what to
do with the JSON you sent.

So `--endpoint-type` still selects **the payload shape and the response parser**
-- it just no longer selects a URL path. Use `chat` for a chat-completions
container, `completions` for a text-completions one, `embeddings` for an
embeddings model. If you pass `--endpoint-type chat` and see no
`/v1/chat/completions` in the logs, that is correct.

## Streaming

`--streaming` switches AIPerf to `InvokeEndpointWithResponseStream`, which is a
genuinely different SageMaker operation on a different path -- not a
content-negotiation flag. AIPerf picks the right one for you.

Getting this wrong manually is quiet rather than loud: calling the non-streaming
path with `"stream": true` in the body returns one buffered response, and your
TTFT and ITL numbers silently become meaningless. Deriving it from `--streaming`
is why you should not hand-write the path.

### Interpreting ITL

The eventstream reader stamps one timestamp per network read and shares it
across every message decoded from that read. This is exactly what AIPerf already
does for SSE, and it is deliberate: frames that arrived in the same TCP segment
genuinely arrived together, so timestamping them individually would only add
decode latency to your measurements and make SageMaker ITL non-comparable to SSE
ITL from the same model.

In other words, ITL from a SageMaker run and ITL from the same vLLM behind plain
HTTP are directly comparable.

## Credentials

AIPerf uses botocore's standard credential chain. If `aws sts get-caller-identity`
works, AIPerf will authenticate.

```bash
# A named profile
aiperf profile -m my-model --sagemaker-endpoint-name my-ep \
    --aws-region us-west-2 --aws-profile my-profile
```

Environment variables, `~/.aws/credentials`, SSO, EC2/ECS instance roles, and
EKS IRSA all work. See the
[SigV4 tutorial](aws-sigv4-auth.md#setting-up-credentials) for the full
walkthrough, including credential rotation during long benchmarks.

### Minimum IAM Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint",
                "sagemaker:InvokeEndpointWithResponseStream"
            ],
            "Resource": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-vllm-endpoint"
        }
    ]
}
```

`InvokeEndpointWithResponseStream` is a separate action from `InvokeEndpoint`.
Granting only the first works until you add `--streaming`, then fails with
`AccessDeniedException`.

## Multi-Model and Multi-Component Endpoints

```bash
# Multi-Model Endpoint: pick which model artifact to invoke
aiperf profile -m my-model --sagemaker-endpoint-name my-mme \
    --aws-region us-west-2 --sagemaker-target-model my-model.tar.gz

# Inference components (SageMaker's newer hosting model)
aiperf profile -m my-model --sagemaker-endpoint-name my-ep \
    --aws-region us-west-2 --sagemaker-inference-component-name my-component

# Pin one production variant, bypassing the endpoint's traffic split
aiperf profile -m my-model --sagemaker-endpoint-name my-ep \
    --aws-region us-west-2 --sagemaker-target-variant variant-b
```

`--sagemaker-target-model` defaults to the `-m` model name, since on a
Multi-Model Endpoint `TargetModel` *is* the model identifier. Note that AWS does
not accept `TargetModel` on the streaming operation at all, so AIPerf omits it
when `--streaming` is set.

`--sagemaker-target-variant` is useful for A/B deployments: it makes every
request land on one variant, so you are benchmarking that variant rather than
the blend.

## VPC / PrivateLink and Custom Domains

An explicit `--url` always wins over the derived hostname, and
`--sagemaker-endpoint-name` still supplies the path:

```bash
aiperf profile -m my-model \
    --sagemaker-endpoint-name my-ep \
    --aws-region us-west-2 \
    --url https://vpce-0abc123-xyz.sagemaker.us-west-2.vpce.amazonaws.com
```

`--aws-region` is still required in this case: it is the SigV4 credential scope,
not just a hostname component.

## What Is Not Supported

Some AIPerf features reach the server over code paths that request signing does
not cover, so their requests would be sent unsigned and rejected. AIPerf refuses
these at startup rather than failing partway into a run:

| Flag | Why |
|---|---|
| `--wait-for-model-timeout` | The readiness probe is an unsigned out-of-band request. |
| `--reset-kv-cache`, server profiler hooks | Control-plane hooks call the server unsigned. |
| Multipart endpoints (`image_edit`, `audio_transcription`) | Streamed multipart bodies never materialize as the bytes SigV4 must hash. |
| Polling endpoints (`video_generation`) | Job submit/poll bypasses signing. |

Also out of scope: **SageMaker Asynchronous Inference** and **Serverless
Inference** endpoints, which exchange payloads through S3 rather than in the HTTP
request body.

## Troubleshooting

### 403 with no detail, or `AccessDeniedException`

Your request went out unsigned or signed wrong. Check, in order:

1. **Is `--aws-region` correct?** It must match the endpoint's region. It sets
   both the hostname and the signing scope.
2. **Do your credentials have `sagemaker:InvokeEndpoint`?** Run
   `aws sagemaker-runtime invoke-endpoint --endpoint-name my-ep ...` to confirm
   independently of AIPerf.
3. **Are you streaming?** That needs the separate
   `sagemaker:InvokeEndpointWithResponseStream` action.
4. **Is your system clock accurate?** AWS rejects signatures more than 5 minutes
   out. Containers and VMs drift; compare `date -u` to real UTC.

### `ValidationError` / `ModelError` from the endpoint

The container rejected your payload. This is almost always a payload-shape
mismatch rather than a SageMaker problem -- check that `--endpoint-type` matches
what the container actually serves.

### "An AWS eventstream response was received but the optional botocore dependency is not installed"

```bash
uv pip install 'aiperf[aws]'
```

### Streaming "works" but TTFT equals total latency

You are hitting the non-streaming path, so the whole response arrives at once.
If you passed an explicit `--url` containing `/invocations`, AIPerf uses it
verbatim; drop the path from `--url` and let `--streaming` select the operation.

### `404` from the endpoint

The endpoint name is wrong, or the endpoint is in a different region than
`--aws-region`. Confirm with:

```bash
aws sagemaker describe-endpoint --endpoint-name my-ep --region us-west-2
```

## See Also

- [AWS SigV4 Authentication](aws-sigv4-auth.md) -- signing against any
  SigV4-protected endpoint, including API Gateway.
- [SageMaker Data Capture](sagemaker-data-capture.md) -- correlating AIPerf
  records with SageMaker's captured inference data. AIPerf sends its request ID
  as `X-Amzn-SageMaker-Inference-Id` on every request, which is the join key.
