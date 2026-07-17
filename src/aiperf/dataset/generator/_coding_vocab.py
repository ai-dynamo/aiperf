# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core vocabulary tuples for CodingContentGenerator.

Extracted from ``coding_content.py`` to stay under the file-size cap.
Contains identifier vocabularies (modules, classes, methods, types,
vars, file paths, routes, error messages, language imports, ML/CUDA
content). Long natural-language tuples live in ``_coding_text.py``.
"""

from __future__ import annotations

# fmt: off
# -- Vocabulary tuples for template fills --

_MODULES = (
    "auth", "cache", "config", "database", "events", "handler", "logger",
    "metrics", "middleware", "pipeline", "processor", "registry", "router",
    "scheduler", "serializer", "service", "storage", "transport", "validator",
    "worker", "adapter", "broker", "collector", "dispatcher", "encoder",
    "factory", "gateway", "indexer", "manager", "monitor", "notifier",
    "observer", "parser", "provider", "queue", "resolver", "scanner",
    "session", "sink", "source", "stream", "transformer", "uploader",
    # web / HTTP
    "api", "webhook", "cors", "oauth", "graphql", "grpc", "websocket",
    "rate_limiter", "proxy", "load_balancer", "reverse_proxy",
    # database / data
    "migration", "schema", "repository", "connection_pool", "query_builder",
    "data_loader", "orm", "replication", "sharding", "backup",
    # ML / data science
    "inference", "tokenizer", "embedding", "feature_store", "model_registry",
    "trainer", "evaluator", "dataset", "sampler", "checkpoint",
    # DevOps / infra
    "deployer", "provisioner", "orchestrator", "health_check", "autoscaler",
    "dns_resolver", "cert_manager", "secret_store", "telemetry", "alerter",
    # security
    "firewall", "encryptor", "key_manager", "audit", "compliance",
    # real libraries / frameworks
    "torch", "numpy", "pandas", "sqlalchemy", "fastapi", "pydantic",
    "celery", "redis", "boto3", "transformers", "datasets", "accelerate",
    "flask", "django", "requests",
)

_CLASSES = (
    "RequestHandler", "DataProcessor", "EventEmitter", "CacheManager",
    "ConnectionPool", "TaskScheduler", "MessageBroker", "StateManager",
    "ConfigLoader", "MetricsCollector", "RateLimiter", "CircuitBreaker",
    "RetryPolicy", "BatchProcessor", "StreamReader", "TokenValidator",
    "SessionStore", "PermissionChecker", "ResourceAllocator", "HealthMonitor",
    "LoadBalancer", "QueueConsumer", "IndexBuilder", "SchemaValidator",
    "PipelineStage", "WorkerPool", "ContextManager", "PluginLoader",
    "TemplateEngine", "SignalHandler", "ProtocolAdapter", "BufferManager",
    "ThrottleController", "RegistryClient", "LockManager", "SnapshotStore",
    "AuditLogger", "FeatureToggle", "MigrationRunner", "DeploymentManager",
    # HTTP layer
    "HttpClient", "RouteResolver", "CorsMiddleware", "AuthMiddleware",
    "ResponseSerializer", "RequestParser", "WebSocketManager", "ApiGateway",
    # data layer
    "QueryExecutor", "TransactionManager", "MigrationEngine", "PoolManager",
    "ReplicaSelector", "ShardRouter", "CursorIterator", "ChangeStream",
    # error types
    "RetryableError", "ValidationError", "TimeoutError", "QuotaExceeded",
    "ConflictError", "NotFoundError", "AuthorizationError", "RateLimitError",
    # ML / inference
    "ModelLoader", "TokenEncoder", "EmbeddingStore", "FeatureExtractor",
    "InferenceEngine", "BatchScheduler", "GradientAccumulator", "Checkpoint",
    # infra / orchestration
    "ServiceMesh", "HealthProbe", "AutoScaler", "SecretProvider",
    "CertRotator", "DnsCache", "TelemetryExporter", "AlertDispatcher",
    # real framework classes
    "Tensor", "DataFrame", "Series", "Session", "Engine", "Router",
    "Pipeline", "Trainer", "Dataset", "DataLoader", "Optimizer",
    "Tokenizer",
)

_METHODS = (
    "process", "handle", "validate", "transform", "execute", "initialize",
    "configure", "dispatch", "resolve", "serialize", "deserialize", "encode",
    "decode", "publish", "subscribe", "notify", "aggregate", "partition",
    "schedule", "allocate", "release", "acquire", "flush", "compress",
    "decompress", "authenticate", "authorize", "revoke", "checkpoint",
    "rollback", "migrate", "replicate", "synchronize", "reconcile",
    "invalidate", "prefetch", "evict", "rebalance", "throttle", "retry",
    "render", "persist", "hydrate", "prune", "drain", "backfill",
    "enqueue", "dequeue", "broadcast", "handshake", "negotiate", "probe",
    "rotate", "shard", "merge", "split", "compact", "snapshot",
    "finalize", "abort", "resume", "suspend", "escalate", "demote",
    "promote", "quarantine", "scrub", "warm_up", "cool_down", "heal",
    "reclaim", "tombstone", "seal", "unseal", "bootstrap", "teardown",
    # real library methods
    "forward", "backward", "train", "evaluate", "predict", "fit",
    "load_state_dict", "save_pretrained", "from_pretrained", "to_dict",
)

_TYPES = (
    "str", "int", "float", "bool", "bytes", "dict", "list", "tuple", "set",
    "None", "Any", "Optional", "Sequence", "Mapping", "Iterator", "Callable",
    "Awaitable", "Coroutine", "AsyncIterator", "Generator", "TypeVar",
    "Protocol", "ClassVar", "Final", "Literal", "Union", "Type",
    "NamedTuple", "TypedDict", "Annotated", "ParamSpec", "Self",
)

_VARS = (
    "result", "data", "config", "context", "payload", "response", "request",
    "buffer", "cursor", "offset", "count", "total", "index", "batch",
    "chunk", "token", "record", "entry", "item", "value", "key", "state",
    "status", "event", "message", "signal", "metric", "timestamp", "duration",
    "timeout", "retries", "threshold", "capacity", "interval", "priority",
    "sequence", "channel", "endpoint", "header", "session", "connection",
    "pipeline", "schema", "trace_id", "tenant_id", "batch_size", "page_size",
    "shard_key", "replica_id", "worker_id", "partition_key", "ttl",
    "max_retries", "backoff", "jitter", "watermark", "checkpoint_id",
    "correlation_id", "span_id", "parent_id", "depth", "fanout",
    "concurrency", "rate", "window", "lag", "drift", "skew",
    "epoch", "generation", "version", "revision", "digest", "nonce",
)

_FILE_PATHS = (
    "src/main.py", "src/config.py", "src/models.py", "src/routes.py",
    "src/utils.py", "src/middleware.py", "src/database.py", "src/auth.py",
    "tests/test_main.py", "tests/test_models.py", "tests/conftest.py",
    "lib/core.go", "lib/handler.go", "lib/service.go", "lib/types.go",
    "pkg/api/server.go", "pkg/api/client.go", "pkg/store/store.go",
    "cmd/server/main.go", "internal/config/config.go",
    "src/lib.rs", "src/main.rs", "src/config.rs", "src/error.rs",
    "src/handler.rs", "src/models.rs", "src/routes.rs",
    "src/index.ts", "src/app.ts", "src/types.ts", "src/api.ts",
    "src/components/App.tsx", "src/components/Form.tsx",
    "Dockerfile", "Makefile", "docker-compose.yml", "pyproject.toml",
    ".github/workflows/ci.yml", "kubernetes/deployment.yaml",
)

_HTTP_ROUTES = (
    "/api/v1/users", "/api/v1/items", "/api/v1/orders", "/api/v1/auth/login",
    "/api/v1/auth/refresh", "/api/v2/search", "/api/v2/analytics",
    "/health", "/ready", "/metrics", "/api/v1/webhooks", "/api/v1/uploads",
    "/api/v1/notifications", "/api/v1/settings", "/api/v1/billing",
    "/api/v1/teams/{team_id}/members", "/api/v1/projects/{project_id}/runs",
    "/api/v1/tenants/{tenant_id}/quota", "/internal/gc", "/internal/debug/pprof",
)

_DB_TABLES = (
    "users", "orders", "items", "sessions", "audit_log", "migrations",
    "api_keys", "rate_limits", "notifications", "webhooks", "tenants",
    "permissions", "invitations", "uploads", "billing_events",
    "job_queue", "dead_letter", "feature_flags", "schema_versions", "locks",
)

_STATUS_CODES = (
    "200 OK", "201 Created", "204 No Content", "301 Moved Permanently",
    "400 Bad Request", "401 Unauthorized", "403 Forbidden", "404 Not Found",
    "409 Conflict", "429 Too Many Requests", "500 Internal Server Error",
    "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout",
)

_LANG_FILE_PATHS: dict[str, tuple[str, ...]] = {
    "python": (
        "src/main.py", "src/config.py", "src/models.py", "src/routes.py",
        "src/utils.py", "src/middleware.py", "src/database.py", "src/auth.py",
        "tests/test_main.py", "tests/test_models.py", "tests/conftest.py",
        "pyproject.toml", "Dockerfile", "Makefile",
        "src/api/v1/endpoints.py", "src/api/v1/schemas.py", "src/api/deps.py",
        "src/core/security.py", "src/core/events.py", "src/services/worker.py",
        "src/repositories/base.py", "tests/integration/test_api.py",
    ),
    "go": (
        "lib/core.go", "lib/handler.go", "lib/service.go", "lib/types.go",
        "pkg/api/server.go", "pkg/api/client.go", "pkg/store/store.go",
        "cmd/server/main.go", "internal/config/config.go",
        "go.mod", "go.sum", "Makefile",
        "internal/middleware/auth.go", "internal/middleware/ratelimit.go",
        "internal/repository/postgres.go", "internal/service/worker.go",
        "pkg/api/middleware.go", "pkg/api/routes.go",
        "internal/telemetry/tracing.go", "internal/health/probe.go",
    ),
    "rust": (
        "src/lib.rs", "src/main.rs", "src/config.rs", "src/error.rs",
        "src/handler.rs", "src/models.rs", "src/routes.rs",
        "Cargo.toml", "Cargo.lock",
        "src/middleware/auth.rs", "src/middleware/tracing.rs",
        "src/repository/mod.rs", "src/repository/postgres.rs",
        "src/service/mod.rs", "src/service/worker.rs",
        "tests/integration/api_test.rs", "benches/throughput.rs",
    ),
    "typescript": (
        "src/index.ts", "src/app.ts", "src/types.ts", "src/api.ts",
        "src/components/App.tsx", "src/components/Form.tsx",
        "src/utils.ts", "src/middleware.ts", "src/routes.ts",
        "package.json", "tsconfig.json", "Dockerfile",
        "src/services/auth.service.ts", "src/services/worker.service.ts",
        "src/middleware/rate-limiter.ts", "src/middleware/error-handler.ts",
        "src/models/user.model.ts", "src/models/order.model.ts",
        "src/repositories/base.repository.ts", "tests/integration/api.test.ts",
    ),
}

_ERROR_MESSAGES = (
    "connection refused", "timeout exceeded", "permission denied",
    "resource not found", "invalid argument", "out of memory",
    "deadlock detected", "rate limit exceeded", "authentication failed",
    "schema validation error", "serialization error", "buffer overflow",
    "index out of range", "null pointer dereference", "type mismatch",
    "missing required field", "duplicate key", "constraint violation",
    "circular dependency detected", "maximum recursion depth exceeded",
    "transaction aborted", "lock timeout after 30s", "quota exceeded",
    "connection pool exhausted", "certificate expired", "DNS resolution failed",
    "checksum mismatch", "payload too large", "stale read",
    "leader election in progress", "shard unavailable", "replica lag exceeded",
    "write conflict detected", "token revoked", "session expired",
    "circuit breaker open", "backpressure applied", "partition offline",
    "consensus timeout", "snapshot corrupted", "migration in progress",
    # GPU / infra errors
    "CUDA out of memory", "NCCL timeout", "connection reset by peer",
    "relation does not exist", "broken pipe", "no route to host",
    "too many open files", "disk quota exceeded",
)

_CLI_COMMANDS = (
    "git status", "git diff HEAD~1", "git log --oneline -10",
    "docker build -t app .", "docker compose up -d",
    "kubectl get pods -n default", "kubectl apply -f deployment.yaml",
    "cargo build --release", "cargo test -- --nocapture",
    "go build ./...", "go test -v ./...", "go vet ./...",
    "npm run build", "npm test", "npx tsc --noEmit",
    "pytest -xvs tests/", "ruff check .", "mypy src/",
    "make build", "make test", "make lint",
    "curl -s http://localhost:8080/health",
    "ps aux | grep python", "top -bn1 | head -20",
    # k8s / infra
    "kubectl describe pod app-7d4b8f-xz9k", "kubectl logs -f deploy/api --tail=100",
    "kubectl rollout status deploy/worker", "kubectl top nodes",
    "helm upgrade --install app ./chart -f values.yaml",
    "terraform plan -out=tfplan", "terraform apply tfplan",
    # redis / data stores
    "redis-cli INFO memory", "redis-cli --latency-history -i 1",
    "pg_dump -Fc mydb > backup.dump", "mongosh --eval 'db.stats()'",
    # perf / profiling
    "perf stat -e cache-misses,cache-references ./bin/server",
    "strace -c -p $(pgrep server)", "valgrind --tool=memcheck ./bin/app",
    "pprof -http=:6060 http://localhost:6060/debug/pprof/heap",
    # load testing
    "wrk -t12 -c400 -d30s http://localhost:8080/api/v1/items",
    "hey -n 10000 -c 100 http://localhost:8080/health",
    "ab -n 5000 -c 50 http://localhost:8080/",
    # misc dev
    "find . -name '*.py' | xargs wc -l | tail -1",
    "du -sh node_modules/ target/ dist/",
    "lsof -i :8080", "ss -tlnp | grep 8080",
    "journalctl -u myapp --since '1 hour ago'",
)

_GO_PACKAGES = (
    "fmt", "os", "io", "net", "http", "context", "sync", "time",
    "strings", "strconv", "encoding/json", "log", "errors", "math",
    "sort", "bytes", "crypto", "regexp", "path/filepath", "database/sql",
    # popular third-party packages
    "github.com/gin-gonic/gin", "go.uber.org/zap",
    "github.com/spf13/viper", "github.com/spf13/cobra",
    "gorm.io/gorm", "google.golang.org/grpc",
    "github.com/prometheus/client_golang/prometheus",
    "github.com/redis/go-redis/v9", "github.com/nats-io/nats.go",
    "github.com/jackc/pgx/v5",
)

_RUST_CRATES = (
    "std::io", "std::fs", "std::collections", "std::sync", "std::fmt",
    "serde", "serde_json", "tokio", "anyhow", "thiserror", "tracing",
    "clap", "reqwest", "axum", "sqlx", "uuid", "chrono", "regex",
    # additional popular crates
    "tower", "hyper", "diesel", "sea_orm", "tonic", "prost",
    "async_trait", "futures",
)

_TS_IMPORTS = (
    "express", "axios", "lodash", "zod", "prisma", "next",
    "react", "react-dom", "typescript", "jest", "vitest",
    "node:fs", "node:path", "node:http", "node:crypto",
    # additional popular packages
    "@nestjs/common", "typeorm", "drizzle-orm", "bullmq",
    "@trpc/server", "ioredis", "pg", "knex",
)

_DECORATORS = (
    "@staticmethod", "@classmethod", "@property", "@abstractmethod",
    "@override", "@cached_property", "@dataclass", "@lru_cache",
    "@pytest.mark.asyncio", "@pytest.mark.parametrize",
    "@app.route", "@app.get", "@app.post", "@router.get",
    # ML framework decorators
    "@torch.no_grad()", "@torch.inference_mode()", "@torch.compile",
    "@torch.jit.script", "@torch.cuda.amp.autocast",
)

_ML_IMPORTS = (
    "torch", "torch.nn", "torch.optim", "torch.utils.data",
    "torch.cuda", "torch.distributed", "torch.amp",
    "transformers", "datasets", "accelerate", "peft",
    "numpy", "safetensors", "wandb", "tensorboard",
    "deepspeed", "bitsandbytes", "trl", "vllm", "triton",
)

_ML_CLASSES = (
    "Linear", "Conv2d", "MultiheadAttention", "LayerNorm", "Embedding",
    "CrossEntropyLoss", "AdamW", "CosineAnnealingLR", "DataLoader",
    "DistributedDataParallel", "AutoModelForCausalLM", "AutoTokenizer",
    "TrainingArguments", "Trainer", "GenerationConfig",
    "BitsAndBytesConfig", "LoraConfig", "PeftModel",
    "StoppingCriteria", "LogitsProcessor",
)

_ML_METHODS = (
    "forward", "backward", "zero_grad", "step", "state_dict",
    "load_state_dict", "save_pretrained", "from_pretrained",
    "generate", "encode", "decode", "batch_decode",
    "to", "cuda", "cpu",
)

_ML_VARS = (
    "logits", "hidden_states", "attention_mask", "input_ids",
    "labels", "loss", "grad_norm", "learning_rate", "num_epochs",
    "batch_size", "max_length", "temperature", "top_p", "top_k",
    "model_name",
)

_MODEL_NAMES = (
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B",
    "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-9b", "Qwen/Qwen2.5-72B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "deepseek-ai/DeepSeek-V3", "microsoft/phi-4",
)

_CUDA_ERRORS = (
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: Expected all tensors to be on the same device",
    "torch.cuda.OutOfMemoryError: CUDA out of memory",
    "NCCL error: unhandled system error, NCCL version 2.18.5",
    "RuntimeError: NCCL communicator was aborted on rank 0",
    "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED",
    "RuntimeError: FlashAttention only supports Ampere GPUs or newer",
    "torch.distributed.DistBackendError: NCCL error",
    "RuntimeError: Deterministic behavior was enabled",
    "CUDA error: device-side assert triggered",
)
# fmt: on
