<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploy and test Riva ASR on GKE

This runbook recreates the real NVIDIA Riva ASR deployment used to validate
AIPerf's native Rust gRPC path. It intentionally contains no credentials. The
deployment is private to the cluster and exposes only `ClusterIP` services.

## Pinned deployment

- Kubernetes context: `nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02`
- Namespace: `acasagrande-riva`
- Helm release: `riva-asr`
- Helm chart: `nvidia-speech/riva-nim` version `1.1.0`
- Image: `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0`
- Model selector: `mode=str`
- Service: `riva-asr`, gRPC `50051`, HTTP/health `9000`
- Hardware: one `NVIDIA-GB200` GPU on `linux/arm64`
- Scheduler/queue: `kai-scheduler`, queue `test`

The official NVIDIA references are the
[Speech NIM Helm guide](https://docs.nvidia.com/nim/speech/latest/deployment/helm.html)
and the
[ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html).

## 1. Authenticate and inspect the cluster

Renew Teleport before using `kubectl`:

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh
kubectl config use-context nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
kubectl cluster-info
```

Confirm the GPU architecture and product rather than assuming them:

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,ARCH:.status.nodeInfo.architecture,GPU:.metadata.labels.nvidia\.com/gpu\.product,COUNT:.status.allocatable.nvidia\.com/gpu'
```

At the time of this deployment, the GPU nodes were Grace/GB200 nodes with both
`kubernetes.io/arch=arm64:NoSchedule` and, on some pools,
`nvidia.com/gpu=present:NoSchedule` taints. The NIM image tag was verified to
publish both `linux/arm64` and `linux/amd64` manifests.

## 2. Create the namespace and NGC secrets

The workstation must already be authenticated to `nvcr.io`. The following
extracts the existing Docker credential into shell variables without printing
it, creates the two secret shapes expected by the chart, and then unsets the
plaintext variables:

```bash
set -euo pipefail

NGC_AUTH="$(jq -er '.auths["nvcr.io"].auth' "$HOME/.docker/config.json")"
NGC_DECODED="$(printf '%s' "$NGC_AUTH" | base64 -d)"
NGC_USER="${NGC_DECODED%%:*}"
NGC_API_KEY="${NGC_DECODED#*:}"
test -n "$NGC_API_KEY"

kubectl create namespace acasagrande-riva

kubectl create secret docker-registry ngc-secret \
  -n acasagrande-riva \
  --docker-server=nvcr.io \
  --docker-username="$NGC_USER" \
  --docker-password="$NGC_API_KEY"

kubectl create secret generic ngc-api \
  -n acasagrande-riva \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"

unset NGC_AUTH NGC_DECODED NGC_USER NGC_API_KEY
```

For an idempotent rerun, add `--dry-run=client -o yaml | kubectl apply -f -`
to each secret command. Never commit the resulting Secret YAML or decoded NGC
key.

## 3. Configure the official Helm repository

Reuse the NGC API key only in the current shell:

```bash
NGC_AUTH="$(jq -er '.auths["nvcr.io"].auth' "$HOME/.docker/config.json")"
NGC_API_KEY="$(printf '%s' "$NGC_AUTH" | base64 -d | cut -d: -f2-)"

helm repo add nvidia-speech https://helm.ngc.nvidia.com/nim/nvidia \
  --username='$oauthtoken' \
  --password="$NGC_API_KEY" \
  --force-update
helm repo update nvidia-speech
helm search repo nvidia-speech/riva-nim --versions

unset NGC_AUTH NGC_API_KEY
```

## 4. Create the deployment values

Write the following to a temporary values file. The `kai.scheduler/queue` label
must be present on the first install because this chart includes `nim.labels`
in the Deployment selector, and selectors are immutable. The namespace's
default KAI queue has zero non-preemptible quota; the cluster's `test` queue is
the intended queue for this canary.

```yaml
image:
  repository: nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us
  tag: "1.5.0"
  pullPolicy: IfNotPresent

imagePullSecrets:
  - name: ngc-secret

nim:
  nimCache: /opt/nim/.cache
  ngcAPISecret: ngc-api
  httpPort: 9000
  grpcPort: 50051
  jsonLogging: true
  logLevel: INFO
  labels:
    kai.scheduler/queue: test

envVars:
  NIM_TAGS_SELECTOR: "mode=str"

replicaCount: 1
statefulSet:
  enabled: false

schedulerName: kai-scheduler
nodeSelector:
  kubernetes.io/arch: arm64
  nvidia.com/gpu.present: "true"
  cloud.google.com/gke-ephemeral-storage-local-ssd: "true"

tolerations:
  - operator: Exists
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: kubernetes.io/arch
    operator: Equal
    value: arm64
    effect: NoSchedule

resources:
  requests:
    cpu: "4"
    memory: 16Gi
    ephemeral-storage: 80Gi
    nvidia.com/gpu: 1
  limits:
    cpu: "16"
    memory: 64Gi
    ephemeral-storage: 80Gi
    nvidia.com/gpu: 1

persistence:
  enabled: false

service:
  name: riva-asr
  type: ClusterIP
  httpPort: 9000
  grpcPort: 50051

startupProbe:
  enabled: true
  method: httpGet
  path: /v1/health/ready
  initialDelaySeconds: 30
  timeoutSeconds: 10
  periodSeconds: 15
  successThreshold: 1
  failureThreshold: 480
```

The cluster admission stack added a 200 GiB scratch allocation to the authored
80 GiB ephemeral-storage request, yielding 280 GiB in the admitted Pod. Keep
the local-SSD selector unless the cluster's scratch policy changes.

## 5. Install and monitor Riva

Render through server-side dry-run before creating resources:

```bash
helm template riva-asr nvidia-speech/riva-nim \
  --version 1.1.0 \
  --namespace acasagrande-riva \
  -f /tmp/acasagrande-riva-values.yaml \
  | kubectl apply --dry-run=server -f -

helm upgrade --install riva-asr nvidia-speech/riva-nim \
  --version 1.1.0 \
  --namespace acasagrande-riva \
  -f /tmp/acasagrande-riva-values.yaml
```

Monitor scheduling, the image pull, RMIR download, TensorRT engine generation,
and readiness:

```bash
kubectl get pods -n acasagrande-riva -o wide
kubectl get events -n acasagrande-riva --sort-by='.lastTimestamp'
kubectl logs -n acasagrande-riva \
  -l app.kubernetes.io/instance=riva-asr \
  --follow
kubectl rollout status deployment/riva-asr-riva-nim \
  -n acasagrande-riva \
  --timeout=30m
```

The first start pulled an approximately 12.8 GB image, downloaded the streaming
Parakeet and punctuation RMIR artifacts, selected the GB200 compute-capability
10.0 profile, and generated GPU-specific TensorRT engines. A refused health
connection on port 9000 is expected until engine generation finishes. A crash,
image-pull error, or terminal model-generation error is not expected.

## 6. Rust protocol-v2 end-to-end test

Build the sole Rust executable from a coherent tree containing commits
`de8a3a981`, `bb33fb417`, `216720dd1`, and compatibility follow-up
`1b398a603`:

```bash
env -u RUSTC_WRAPPER cargo build --manifest-path rust/Cargo.toml --locked --release -p aiperf-cli
rust/target/release/aiperf --capabilities \
  | jq '{distribution_id, endpoint_types, supported_pairs}'
```

The capabilities must include `riva_asr` and the
`["online_grpc", "scheduled"]` pair.

Forward the private service locally:

```bash
kubectl port-forward -n acasagrande-riva service/riva-asr 50051:50051
```

Prepare real 16 kHz mono speech. The common ALSA fixture says “front center”:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -i /usr/share/sounds/alsa/Front_Center.wav \
  -ac 1 -ar 16000 -c:a pcm_s16le \
  /tmp/riva-front-center-16k.wav
```

Use the exact same payload for authored validation and execution, changing only
the operation. The model resource name below is the name generated by this
streaming RMIR profile:

```bash
set -euo pipefail

RUNNER=rust/target/release/aiperf
TARGET="/tmp/aiperf-riva-gke-real-$(date +%s)"
DIST="$($RUNNER --capabilities | jq -er '.distribution_id')"
AUDIO="$(base64 -w0 /tmp/riva-front-center-16k.wav)"

make_request() {
  local operation="$1"
  jq -cn \
    --arg operation "$operation" \
    --arg distribution_id "$DIST" \
    --arg target "$TARGET" \
    --arg url grpc://127.0.0.1:50051 \
    --arg audio "$AUDIO" \
    '{
      protocol_version: 2,
      operation: $operation,
      expected_distribution_id: $distribution_id,
      run: {
        identity: {
          benchmark_id: "gke-riva-asr-real",
          random_seed: 11
        },
        artifact_target: $target,
        backend: {type: "online_grpc", config: {}},
        workload: {
          type: "scheduled",
          config: {
            worker_count: 1,
            dataset: {
              type: "file",
              format: "single_turn",
              sampling: "sequential",
              records: [{
                audio: ("wav," + $audio),
                output_length: 8
              }]
            },
            tokenizer: {
              name: "cl100k_base",
              revision: "main",
              trust_remote_code: false,
              apply_chat_template: false
            },
            phases: [{
              type: "concurrency",
              name: "profiling",
              exclude_from_results: false,
              requests: 1,
              concurrency: 1
            }]
          }
        },
        resources: {
          models: {
            strategy: "round_robin",
            items: [{
              name: "parakeet-1.1b-en-US-asr-streaming"
            }]
          },
          endpoints: {
            profiles: [{
              id: "default",
              type: "riva_asr",
              urls: [$url],
              streaming: true,
              use_server_token_count: false,
              timeout_seconds: 120.0,
              connection_reuse: "pooled",
              headers: {},
              extra: {
                language_code: "en-US",
                sample_rate_hertz: 16000,
                encoding: "LINEAR_PCM",
                chunk_size: 8000
              },
              http2: false,
              wait_for_model_timeout: 0.0
            }]
          },
          metrics: {},
          artifacts: {},
          sidecars: {}
        }
      }
    }'
}

test ! -e "$TARGET"
make_request validate | "$RUNNER"
test ! -e "$TARGET"  # validation is side-effect free

make_request execute | "$RUNNER"
jq '.run, .metrics.request_count' "$TARGET/native-v2.json"
```

The authored validation performed during the first deployment returned
`success: true`, `completeness: static`, and left the artifact target absent.

## Live execution results (verified 2026-07-12)

Once the Pod reached `1/1 Ready`, the strict protocol-v2 `execute` operation was
run against the live service through the local port-forward. It succeeded:

```
{"protocol_version":2,"event":"run_terminal","success":true,
 "provenance":{"backend":"online_grpc","transport":"grpc","workload":"scheduled"}}
```

The canonical NVIDIA sample audio bundled inside the NIM
(`/opt/riva/wav/en-US_sample.wav`, 16 kHz mono, "what is natural language
processing") was copied out of the Pod and used as the input. Enabling the
optional record artifacts confirms the native Rust gRPC path decodes the real
streaming transcript byte-for-byte:

```bash
# add to run.resources.artifacts to capture the transcript
"artifacts": {"records_path": "profile_export.jsonl",
              "raw_path": "profile_export_raw.jsonl"}
```

Observed for the sample audio:

- 18 client messages sent (1 `StreamingRecognitionConfig` + 17 audio chunks),
  26 `StreamingRecognizeResponse` messages received.
- Final transcript in `profile_export_raw.jsonl`:
  `What is natural language processing? ` (matches Riva's own
  `transcribe_file.py` reference client run against the same server).
- `request_latency` ≈ 540 ms, `request_error_rate` = 0, one completed request.

### Endpoint coverage summary — all 9 verified end-to-end (2026-07-12)

Every one of the runner's nine Riva endpoints was exercised against a **real**
Riva server and returned correct output. ASR + TTS via NIMs on the GKE GB200;
the six NLP RPCs via a locally-run classic Riva 2.13 server (see
"The five NLP RPCs" below).

| # | Endpoint | Server | Real output |
|---|---|---|---|
| 1 | `riva_asr` | ASR NIM (GKE) | `What is natural language processing?` |
| 2 | `riva_tts` | Magpie TTS NIM (GKE) | 110592 audio samples (2.51 s) |
| 3 | `riva_text_classify` | classic 2.13 (local) | `meteorology` (0.985) |
| 4 | `riva_token_classify` | classic 2.13 (local) | PER/ORG/LOC NER tokens |
| 5 | `riva_analyze_entities` | classic 2.13 (local) | same NER tokens |
| 6 | `riva_analyze_intent` | classic 2.13 (local) | `weather.weather` (0.997) + slots |
| 7 | `riva_natural_query` | classic 2.13 (local) | answer `27` (0.955) |
| 8 | `riva_transform_text` | classic 2.13 (local) | punctuated + capitalized |
| 9 | `riva_punctuate_text` | ASR NIM + classic (local) | punctuated + capitalized |

### Endpoint coverage against the ASR-NIM deployment

The `parakeet-1-1b-ctc-en-us` NIM registers exactly two Triton models —
`parakeet-1.1b-en-US-asr-streaming` (ASR) and `riva-punctuation-en-US` (NLP) —
so only the endpoints those two models serve can be exercised here. The runner
exposes nine Riva endpoints; results against this deployment:

| Runner endpoint | RPC | Model here | Result |
|---|---|---|---|
| `riva_asr` | `StreamingRecognize` | parakeet ASR | ✅ `What is natural language processing?` |
| `riva_punctuate_text` | `PunctuateText` | riva-punctuation | ✅ `hello…` → capitalized + punctuated |
| `riva_transform_text` | `TransformText` | (none) | gRPC 12 UNIMPLEMENTED — server has the method but no model wired to it here; runner path is correct |
| `riva_tts` | `Synthesize`/`SynthesizeOnline` | — | needs a Riva TTS NIM |
| `riva_text_classify` | `ClassifyText` | — | needs an NLP classification model |
| `riva_token_classify` | `ClassifyTokens` | — | needs an NLP NER model |
| `riva_analyze_intent` | `AnalyzeIntent` | — | needs an NLP intent/slot model |
| `riva_analyze_entities` | `AnalyzeEntities` | — | needs an NLP entities model |
| `riva_natural_query` | `NaturalQuery` | — | needs an NLP QA model |

To exercise the remaining six, deploy a Riva TTS NIM and the corresponding NLP
NIMs (each is a separate ~12 GB GPU model with its own first-start generation).
The `RivaLanguageUnderstanding` service on this server does expose all method
names (verified via `riva_nlp_pb2` `services_by_name`), so UNIMPLEMENTED for
`TransformText` is a missing-model condition, not a wrong method path.

Text NLP endpoints take a `single_turn` dataset with a `text` field (not
`prompt`) and a non-streaming profile; set `resources.models.items[].name` to
the served NLP model.

Notes learned while validating:

- `speech_duration:0.0` in the server-side `stats_builder` cloud event is
  **not** an error and does not mean "no speech"; the NIM leaves that field at
  zero even for the reference client's successful transcription. Trust the
  decoded transcript, not that field.
- `riva_asr` sets `produces_tokens: false`, so `output_sequence_length` /
  token-throughput read `0` for ASR runs. The transcript is still captured in
  `response_text`; ASR fidelity is the transcript and latency, not token count.
- The ALSA `Front_Center.wav` fixture transcribes to an empty string on this
  model — it is a short chime-like clip, not clear speech. Use the NIM's own
  `/opt/riva/wav/en-US_sample.wav` (or any real speech) for a non-empty result.
- To reproduce the ground-truth transcript directly on the server, exec into the
  Pod and run
  `python3 /opt/riva/examples/transcribe_file.py --server localhost:50051 --input-file /opt/riva/wav/en-US_sample.wav --language-code en-US`.

## Exercising the other endpoints: swap the model on the same GPU

The cluster has one GB200 GPU in the `test` queue, so each additional endpoint
family is covered by **uninstalling the current release and installing a
compatible NIM in its place** (same chart, different image + `NIM_TAGS_SELECTOR`,
same `service.grpcPort: 50051`). Reuse the namespace and NGC secrets across
swaps; only the Helm release changes.

```bash
helm uninstall <current-release> -n acasagrande-riva
kubectl wait --for=delete pod -n acasagrande-riva -l app.kubernetes.io/instance=<current-release> --timeout=120s
helm upgrade --install <new-release> nvidia-speech/riva-nim --version 1.1.0 \
  --namespace acasagrande-riva -f /tmp/<new-values>.yaml
```

### Architecture gotcha (GB200 is arm64)

Not every Speech NIM ships an arm64 image. Check before pulling 12 GB:

```bash
docker manifest inspect nvcr.io/nim/nvidia/<image>:<tag> \
  | jq -r '.manifests[]?.platform | "\(.os)/\(.architecture)"' | sort -u
```

Observed on 2026-07-12:
- `parakeet-1-1b-ctc-en-us:1.5.0` (ASR) → amd64 **and** arm64 ✅
- `riva-tts:1.3.0` (FastPitch HiFi-GAN TTS) → **amd64 only** ❌ — will not
  schedule on the arm64 GB200 node.
- `magpie-tts-multilingual:1.8.0` (TTS) → amd64 **and** arm64 ✅ — this is the
  TTS NIM to use on Grace/Blackwell. (`magpie-tts-zeroshot` / `magpie-tts-flow`
  require NGC access approval and returned no pullable tags.)

List tags without the `ngc` CLI via the NGC registry token exchange:

```bash
NGC_API_KEY="$(jq -r '.auths["nvcr.io"].auth' ~/.docker/config.json | base64 -d | cut -d: -f2-)"
TOKEN="$(curl -s -u "\$oauthtoken:$NGC_API_KEY" "https://nvcr.io/proxy_auth?scope=repository:nim/nvidia/<image>:pull" | jq -r '.token')"
curl -s -H "Authorization: Bearer $TOKEN" "https://nvcr.io/v2/nim/nvidia/<image>/tags/list" | jq -r '.tags[]?'
```

### TTS: `riva_tts` (Magpie TTS Multilingual)

Values differ from ASR only in the image, selector, and service name:

```yaml
image:
  repository: nvcr.io/nim/nvidia/magpie-tts-multilingual
  tag: "1.8.0"
envVars:
  NIM_TAGS_SELECTOR: "name=magpie-tts-multilingual"
service:
  name: riva-tts
# nodeSelector / tolerations / resources / kai queue label identical to ASR
```

Deploy: `helm upgrade --install riva-tts nvidia-speech/riva-nim --version 1.1.0
-n acasagrande-riva -f /tmp/acasagrande-tts-values.yaml`. On CC>=10 (GB200) the
NIM defaults to `batch_size=1`.

Live `riva_tts` result (verified 2026-07-12): input
`The quick brown fox jumps over the lazy dog.` with voice
`Magpie-Multilingual.EN-US.Aria` synthesized **110592 audio samples**
(`audio_duration` 2.51 s, `total_characters` 44, server `status:0`), and the
runner completed with `status:200` / `grpc-status:0`. List valid voices with
`kubectl exec <tts-pod> -- curl -s http://localhost:9000/v1/audio/list_voices`.
The `riva_tts` endpoint payload takes `extra.voice_name`, `language_code`,
`encoding`, `sample_rate_hz` and the turn `text`.

### The five NLP RPCs: no NIM — use the classic Riva Speech Skills server

`ClassifyText`, `ClassifyTokens`, `AnalyzeIntent`, `AnalyzeEntities`,
`NaturalQuery`, and `TransformText` have **no NIM** (Speech NIM catalog is
ASR + TTS + NMT only; probing `nvcr.io/nim/nvidia/riva-nlp*` etc. returns no
pullable tags). They ship only in the classic **Riva Speech Skills** server. The
`riva-punctuation-en-US` model that answered `riva_punctuate_text` on the ASR NIM
was **bundled inside that NIM**, not a standalone deployment.

These were exercised end-to-end by running the classic Riva server **locally on
an x86_64 workstation with an RTX 6000 Ada** (Docker + GPU). The GKE GB200 nodes
are arm64 and the classic pre-built NLP models are x86-only, so local x86 is the
practical target.

**Procedure (verified 2026-07-12, all six NLP RPCs returned correct output):**

1. NGC auth. The workstation's `~/.docker/config.json` holds an `nvapi-` key that
   works as a direct bearer for the NGC resource API and with the `ngc` CLI:
   ```bash
   NGC_API_KEY="$(jq -r '.auths["nvcr.io"].auth' ~/.docker/config.json | base64 -d | cut -d: -f2-)"
   export NGC_CLI_API_KEY="$NGC_API_KEY" NGC_CLI_ORG=nvidia NGC_CLI_TEAM=riva
   ```

2. **Use a version-matched Riva**, not the latest. The Apr-2023 `deployable_v1.0`
   NLP models do **not** deploy under the 2.19.0 servicemaker — its config
   generation cascades `AttributeError: '<Model>' object has no attribute
   'input_ids_type' / entity_count / qa_dims / _inputs` because `set_bert_args`
   never runs for that model format. Riva **2.13.0** matches. Pull the
   `-servicemaker` image (the plain `riva-speech:2.13.0` server image has **no**
   `riva-build`/`riva-deploy`):
   ```bash
   docker pull nvcr.io/nvidia/riva/riva-speech:2.13.0
   docker pull nvcr.io/nvidia/riva/riva-speech:2.13.0-servicemaker
   ```

3. Download the **pre-built 2.13.0 RMIRs** (no `riva-build` needed) via `ngc`:
   ```bash
   for m in rmir_nlp_named_entity_recognition_bert_base rmir_nlp_intent_slot_bert_base \
            rmir_nlp_question_answering_bert_base rmir_nlp_text_classification_bert_base \
            rmir_nlp_punctuation_bert_base_en_us; do
     ngc registry model download-version "nvidia/riva/$m:2.13.0" --dest /tmp/riva-nlp/dl
   done   # gives ner_default / intent_slot_weather / qa_bertbase / tc_misty / punctuation .rmir
   ```

4. `riva-deploy` each RMIR (key `tlt_encode`) into a model repo with the
   **matched servicemaker** (needs `--gpus` for the TensorRT build):
   ```bash
   docker run --rm --gpus '"device=0"' -v /tmp/riva-nlp:/data \
     nvcr.io/nvidia/riva/riva-speech:2.13.0-servicemaker bash -lc '
       for f in ner_default intent_slot_weather qa_bertbase tc_misty punct; do
         riva-deploy /data/rmir/${f}.rmir:tlt_encode /data/models/
       done'
   ```
   Registers `riva_ner`, `riva_intent_weather`, `riva_qa`,
   `riva_text_classification_domain`, `riva-punctuation-en-US`.

5. Serve locally with the 2.13.0 server image:
   ```bash
   docker run -d --name riva-nlp-server --gpus '"device=0"' \
     -v /tmp/riva-nlp:/data -p 50051:50051 \
     nvcr.io/nvidia/riva/riva-speech:2.13.0 \
     start-riva --riva-uri=0.0.0.0:50051 --nlp_service=true --asr_service=false --tts_service=false
   ```

6. Point the runner at `grpc://127.0.0.1:50051` (no port-forward — it's local).
   Endpoint → `resources.models.items[].name` → input:

   | Runner endpoint | model name | input | Verified output |
   |---|---|---|---|
   | `riva_text_classify` | `riva_text_classification_domain` | `text` | `meteorology` (0.985) |
   | `riva_token_classify` | `riva_ner` | `text` | jensen huang=PER, nvidia=ORG, santa clara=LOC |
   | `riva_analyze_entities` | `riva_ner` | `text` | same NER token labels |
   | `riva_analyze_intent` | `riva_intent_weather` | `text` + `extra.domain:"weather"` | intent `weather.weather` (0.997) + slots |
   | `riva_natural_query` | `riva_qa` | `text` + `extra.context:"…"` | answer `27` (0.955) |
   | `riva_transform_text` | `riva-punctuation-en-US` | `text` | punctuated + capitalized |
   | `riva_punctuate_text` | `riva-punctuation-en-US` | `text` | punctuated + capitalized |

   Note: the classic punctuation model serves **both** `TransformText` and
   `PunctuateText`; the ASR NIM's bundled punctuation served only `PunctuateText`
   (so `riva_transform_text` returns gRPC 12 UNIMPLEMENTED against the NIM — that
   is a per-deployment model-wiring difference, not a runner bug).

**Container gotchas hit while deploying (all version-mismatch symptoms — avoided
by using 2.13.0, listed here in case a future task needs a newer base):**
- `riva-deploy` calls `python` (not `python3`) and hardcodes a `python3.10`
  package path. On a `python3.12` image add
  `ln -sf "$(which python3)" /usr/bin/python` and
  `ln -sfn /usr/local/lib/python3.12 /usr/local/lib/python3.10`.
- `riva-deploy -f` takes a directory only in newer builds; older ones want
  explicit `file.rmir:tlt_encode` arguments.
- Model-repo dirs are written as root; clean them via a throwaway container
  (`docker run --rm -v /tmp/riva-nlp:/data alpine rm -rf /data/models`).

## Troubleshooting learned during the first deployment

- `NonPreemptibleOverQuota ... default-queue quota is 0`: ensure
  `nim.labels.kai.scheduler/queue: test` is present before the first install.
- `spec.selector ... field is immutable`: `nim.labels` changed after install.
  Uninstall only the failed Helm release and reinstall with the final labels;
  retain the namespace and NGC secrets.
- `ContainerCreating` for about 90 seconds: the 12.8 GB image is still pulling.
- Startup probe connection refused while logs show `riva-deploy` or TensorRT
  compilation: expected first-start behavior; the probe budget above permits up
  to two hours.
- Do not use the current Riva SDK Helm chart on these data-center nodes. Current
  SDK releases target embedded L4T; Speech NIM is the data-center deployment
  surface and publishes a native ARM64 image for this model.

## Cleanup

Remove the service while retaining the namespace credentials:

```bash
helm uninstall riva-asr -n acasagrande-riva
```

Remove everything, including NGC secrets:

```bash
kubectl delete namespace acasagrande-riva
```
