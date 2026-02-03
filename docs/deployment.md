<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide

This guide covers deploying AIPerf in various environments from local Docker to production Kubernetes clusters.

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Multi-Node Setups](#multi-node-setups)
- [Configuration Management](#configuration-management)
- [Security Considerations](#security-considerations)
- [Monitoring Integration](#monitoring-integration)
- [CI/CD Integration](#cicd-integration)
- [Cloud Provider Examples](#cloud-provider-examples)

## Docker Deployment

### Basic Docker Usage

Run AIPerf as a container:

```bash
# Pull official image
docker pull aidynamo/aiperf:latest

# Run a benchmark
docker run --rm --network=host \
  aidynamo/aiperf:latest \
  profile \
    --model Qwen/Qwen3-0.6B \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-count 100
```

### Saving Results from Container

Mount a volume to persist output:

```bash
docker run --rm \
  --network=host \
  -v $(pwd)/artifacts:/workspace/artifacts \
  aidynamo/aiperf:latest \
  profile \
    --model Qwen/Qwen3-0.6B \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --request-count 100 \
    --output-artifact-dir /workspace/artifacts
```

Results will be saved to `./artifacts/` on your host.

### Using Custom Datasets

Mount dataset files into the container:

```bash
docker run --rm \
  --network=host \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/artifacts:/workspace/artifacts \
  aidynamo/aiperf:latest \
  profile \
    --input-file /data/prompts.jsonl \
    --custom-dataset-type single_turn \
    --url http://localhost:8000 \
    --endpoint-type chat
```

### Environment Variables

Pass environment variables for configuration:

```bash
docker run --rm \
  --network=host \
  -e AIPERF_HTTP_CONNECTION_LIMIT=5000 \
  -e AIPERF_WORKER_MAX_WORKERS_CAP=32 \
  aidynamo/aiperf:latest \
  profile --model your_model --url http://localhost:8000
```

Or use an env file:

```bash
# Create aiperf.env
cat > aiperf.env <<EOF
AIPERF_HTTP_CONNECTION_LIMIT=5000
AIPERF_WORKER_MAX_WORKERS_CAP=32
AIPERF_HTTP_SO_RCVBUF=10485760
EOF

# Use with --env-file
docker run --rm --network=host --env-file aiperf.env \
  aidynamo/aiperf:latest profile ...
```

### Docker Compose

Create a `docker-compose.yml` for repeatable deployments:

```yaml
version: '3.8'

services:
  # Inference server (example with vLLM)
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen3-0.6B
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # AIPerf benchmark
  aiperf:
    image: aidynamo/aiperf:latest
    command: >
      profile
      --model Qwen/Qwen3-0.6B
      --url http://vllm:8000
      --endpoint-type chat
      --streaming
      --concurrency 10
      --request-count 100
    depends_on:
      - vllm
    volumes:
      - ./artifacts:/workspace/artifacts
    environment:
      - AIPERF_HTTP_CONNECTION_LIMIT=5000
```

Run with:
```bash
docker-compose up
```

### Building Custom Images

If you need custom plugins or configurations:

```dockerfile
# Dockerfile.custom
FROM aidynamo/aiperf:latest

# Install custom dependencies
RUN pip install your-custom-plugin

# Copy custom plugins
COPY plugins/ /app/plugins/
COPY plugins.yaml /app/plugins.yaml

# Set environment defaults
ENV AIPERF_HTTP_CONNECTION_LIMIT=5000
ENV AIPERF_WORKER_MAX_WORKERS_CAP=32
```

Build and use:
```bash
docker build -f Dockerfile.custom -t aiperf-custom:latest .
docker run --rm --network=host aiperf-custom:latest profile ...
```

## Kubernetes Deployment

### Basic Kubernetes Job

Deploy AIPerf as a Kubernetes Job:

```yaml
# aiperf-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: aiperf-benchmark
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: aiperf
        image: aidynamo/aiperf:latest
        command:
        - aiperf
        - profile
        - --model
        - Qwen/Qwen3-0.6B
        - --url
        - http://inference-service:8000
        - --endpoint-type
        - chat
        - --streaming
        - --concurrency
        - "50"
        - --request-count
        - "1000"
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
          limits:
            cpu: "8"
            memory: "16Gi"
        volumeMounts:
        - name: artifacts
          mountPath: /workspace/artifacts
      volumes:
      - name: artifacts
        persistentVolumeClaim:
          claimName: aiperf-artifacts-pvc
```

Apply with:
```bash
kubectl apply -f aiperf-job.yaml
```

### Persistent Volume for Results

Create a PVC for storing results:

```yaml
# artifacts-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: aiperf-artifacts-pvc
spec:
  accessModes:
    - ReadWriteMany  # Allow multiple pods to write
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd  # Use your storage class
```

### ConfigMap for Configuration

Store configuration in a ConfigMap:

```yaml
# aiperf-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aiperf-config
data:
  AIPERF_HTTP_CONNECTION_LIMIT: "5000"
  AIPERF_WORKER_MAX_WORKERS_CAP: "32"
  AIPERF_HTTP_SO_RCVBUF: "10485760"
  AIPERF_HTTP_SO_SNDBUF: "10485760"
```

Reference in Job:
```yaml
spec:
  template:
    spec:
      containers:
      - name: aiperf
        envFrom:
        - configMapRef:
            name: aiperf-config
```

### Custom Datasets in Kubernetes

Mount datasets from ConfigMap or PVC:

```yaml
# datasets-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: aiperf-datasets-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
```

In Job:
```yaml
spec:
  template:
    spec:
      containers:
      - name: aiperf
        command:
        - aiperf
        - profile
        - --input-file
        - /datasets/prompts.jsonl
        - --custom-dataset-type
        - single_turn
        volumeMounts:
        - name: datasets
          mountPath: /datasets
          readOnly: true
      volumes:
      - name: datasets
        persistentVolumeClaim:
          claimName: aiperf-datasets-pvc
```

### CronJob for Scheduled Benchmarks

Run benchmarks on a schedule:

```yaml
# aiperf-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: aiperf-daily-benchmark
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: aiperf
            image: aidynamo/aiperf:latest
            command:
            - /bin/bash
            - -c
            - |
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)
              aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url http://inference-service:8000 \
                --endpoint-type chat \
                --streaming \
                --concurrency 50 \
                --request-count 1000 \
                --random-seed 42 \
                --profile-export-prefix benchmark_${TIMESTAMP} \
                --output-artifact-dir /artifacts/${TIMESTAMP}
            volumeMounts:
            - name: artifacts
              mountPath: /artifacts
          volumes:
          - name: artifacts
            persistentVolumeClaim:
              claimName: aiperf-artifacts-pvc
```

### Distributed Benchmarking

Run multiple AIPerf pods for higher throughput:

```yaml
# aiperf-distributed.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: aiperf-distributed
spec:
  parallelism: 4  # Run 4 pods in parallel
  completions: 4
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: aiperf
        image: aidynamo/aiperf:latest
        command:
        - aiperf
        - profile
        - --model
        - Qwen/Qwen3-0.6B
        - --url
        - http://inference-service:8000
        - --endpoint-type
        - chat
        - --concurrency
        - "25"  # 25 per pod = 100 total
        - --request-count
        - "250"  # 250 per pod = 1000 total
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
```

**Note:** Results from each pod are independent. Aggregate manually.

## Multi-Node Setups

### Client-Side Load Balancing

Use AIPerf's multi-URL feature:

```bash
aiperf profile \
  --url http://node1:8000 \
  --url http://node2:8000 \
  --url http://node3:8000 \
  --url-strategy round_robin \
  --workers-max 64 \
  --concurrency 300
```

This distributes load across multiple servers from a single AIPerf instance.

### Multiple AIPerf Instances

Run separate AIPerf instances against different servers:

```bash
# Terminal 1 - Node 1
aiperf profile --url http://node1:8000 --concurrency 100 \
  --profile-export-prefix node1

# Terminal 2 - Node 2
aiperf profile --url http://node2:8000 --concurrency 100 \
  --profile-export-prefix node2

# Terminal 3 - Node 3
aiperf profile --url http://node3:8000 --concurrency 100 \
  --profile-export-prefix node3
```

### Load Balancer Configuration

Use an external load balancer:

```bash
# AIPerf → Load Balancer → Multiple Servers
aiperf profile \
  --url http://loadbalancer:8000 \
  --workers-max 64 \
  --concurrency 500
```

**Load balancer considerations:**
- Use least-connections or round-robin algorithm
- Enable session affinity if benchmarking multi-turn conversations
- Set appropriate timeouts (match `--request-timeout-seconds`)

### Shared Dataset Storage

For Kubernetes distributed deployments with shared datasets:

```yaml
# Use ReadWriteMany volume for dataset sharing
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-datasets
spec:
  accessModes:
    - ReadWriteMany  # Multiple pods can read
  resources:
    requests:
      storage: 50Gi
```

Configure AIPerf to use shared storage:

```bash
# Via environment variable
export AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-datasets

# Kubernetes ConfigMap
AIPERF_DATASET_MMAP_BASE_PATH: "/mnt/shared-datasets"
```

## Configuration Management

### Environment-Specific Configurations

Use separate configs for dev, staging, prod:

```bash
# configs/dev.env
AIPERF_HTTP_CONNECTION_LIMIT=1000
AIPERF_WORKER_MAX_WORKERS_CAP=8

# configs/staging.env
AIPERF_HTTP_CONNECTION_LIMIT=5000
AIPERF_WORKER_MAX_WORKERS_CAP=16

# configs/prod.env
AIPERF_HTTP_CONNECTION_LIMIT=10000
AIPERF_WORKER_MAX_WORKERS_CAP=32
```

Use with:
```bash
docker run --rm --env-file configs/prod.env aidynamo/aiperf:latest profile ...
```

### Secrets Management

For API keys and sensitive data:

**Kubernetes Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aiperf-secrets
type: Opaque
stringData:
  api-key: "your-secret-api-key"
```

Reference in Pod:
```yaml
spec:
  containers:
  - name: aiperf
    env:
    - name: API_KEY
      valueFrom:
        secretKeyRef:
          name: aiperf-secrets
          key: api-key
    command:
    - aiperf
    - profile
    - --api-key
    - $(API_KEY)
```

**Docker:**
```bash
# Use environment variable
export API_KEY="your-secret-key"
docker run --rm -e API_KEY aidynamo/aiperf:latest \
  profile --api-key $API_KEY ...
```

## Security Considerations

### Network Policies

Restrict AIPerf network access in Kubernetes:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aiperf-netpol
spec:
  podSelector:
    matchLabels:
      app: aiperf
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: inference-server
    ports:
    - protocol: TCP
      port: 8000
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

### SSL/TLS

Enable SSL verification for production:

```bash
# Default: SSL verification enabled
aiperf profile --url https://server:8443 ...

# Disable only for testing (NOT for production)
export AIPERF_HTTP_SSL_VERIFY=False
```

### Resource Limits

Set appropriate limits in Kubernetes:

```yaml
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "8"      # Prevents CPU starvation of other pods
    memory: "16Gi"  # Prevents OOM kills
```

### Read-Only Filesystems

Run with read-only root filesystem:

```yaml
spec:
  containers:
  - name: aiperf
    securityContext:
      readOnlyRootFilesystem: true
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: artifacts
      mountPath: /workspace/artifacts
  volumes:
  - name: tmp
    emptyDir: {}
```

## Monitoring Integration

### Prometheus Metrics

Export AIPerf results to Prometheus:

```bash
# Run benchmark and export
aiperf profile ... --export-level records

# Parse and push to Prometheus Pushgateway
python scripts/push_to_prometheus.py \
  artifacts/*/profile_export.json \
  http://pushgateway:9091
```

### Grafana Dashboards

Import results into Grafana for visualization:

1. Export results as JSON
2. Use Grafana's JSON datasource
3. Create dashboards for key metrics (TTFT, throughput, etc.)

### Log Aggregation

Send AIPerf logs to centralized logging:

**Kubernetes with Fluentd:**
```yaml
spec:
  containers:
  - name: aiperf
    command:
    - aiperf
    - profile
    - --log-level
    - INFO
  # Fluentd sidecar automatically collects logs
```

**Docker with log driver:**
```bash
docker run --rm \
  --log-driver=fluentd \
  --log-opt fluentd-address=localhost:24224 \
  aidynamo/aiperf:latest profile ...
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/benchmark.yml
name: AIPerf Benchmark

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Start inference server
      run: |
        docker run -d --name vllm -p 8000:8000 \
          vllm/vllm-openai:latest \
          --model Qwen/Qwen3-0.6B

        # Wait for server
        timeout 300 bash -c 'until curl -s http://localhost:8000/health; do sleep 2; done'

    - name: Run benchmark
      run: |
        docker run --rm --network=host \
          -v $PWD/artifacts:/workspace/artifacts \
          aidynamo/aiperf:latest \
          profile \
            --model Qwen/Qwen3-0.6B \
            --url http://localhost:8000 \
            --endpoint-type chat \
            --streaming \
            --concurrency 10 \
            --request-count 100 \
            --random-seed 42

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: artifacts/

    - name: Compare to baseline
      run: |
        python scripts/compare_baseline.py \
          artifacts/*/profile_export.json \
          baseline.json \
          --threshold 0.10
```

### GitLab CI Example

```yaml
# .gitlab-ci.yml
benchmark:
  stage: test
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker pull vllm/vllm-openai:latest
    - docker run -d --name vllm -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen3-0.6B
    - sleep 60  # Wait for server
    - docker run --rm --network=host -v $PWD/artifacts:/workspace/artifacts
        aidynamo/aiperf:latest profile
        --model Qwen/Qwen3-0.6B
        --url http://localhost:8000
        --endpoint-type chat
        --concurrency 10
        --request-count 100
  artifacts:
    paths:
      - artifacts/
    expire_in: 30 days
```

## Cloud Provider Examples

### AWS ECS

```json
{
  "family": "aiperf-task",
  "containerDefinitions": [
    {
      "name": "aiperf",
      "image": "aidynamo/aiperf:latest",
      "command": [
        "profile",
        "--model", "Qwen/Qwen3-0.6B",
        "--url", "http://inference-server:8000",
        "--endpoint-type", "chat",
        "--concurrency", "50",
        "--request-count", "1000"
      ],
      "cpu": 4096,
      "memory": 8192,
      "mountPoints": [
        {
          "sourceVolume": "artifacts",
          "containerPath": "/workspace/artifacts"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "artifacts",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678"
      }
    }
  ]
}
```

### GCP Cloud Run Job

```yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: aiperf-benchmark
spec:
  template:
    spec:
      containers:
      - name: aiperf
        image: aidynamo/aiperf:latest
        args:
        - profile
        - --model
        - Qwen/Qwen3-0.6B
        - --url
        - http://inference-service
        - --endpoint-type
        - chat
        - --concurrency
        - "50"
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
```

### Azure Container Instances

```yaml
apiVersion: '2021-09-01'
location: eastus
name: aiperf-benchmark
properties:
  containers:
  - name: aiperf
    properties:
      image: aidynamo/aiperf:latest
      command:
      - aiperf
      - profile
      - --model
      - Qwen/Qwen3-0.6B
      - --url
      - http://inference-service:8000
      - --endpoint-type
      - chat
      - --concurrency
      - "50"
      resources:
        requests:
          cpu: 4
          memoryInGb: 8
  osType: Linux
```

## See Also

- **[Getting Started](getting-started.md)** - Installation basics
- **[Performance Tuning](performance-tuning.md)** - Optimize for scale
- **[Configuration](environment_variables.md)** - Environment variables
- **[Best Practices](best-practices.md)** - Deployment guidelines
- **[Troubleshooting](troubleshooting.md)** - Common deployment issues
