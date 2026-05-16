# Containers-mode pod: N sibling containers, each a standalone worker.
# Each container has its own Python interpreter, its own private import heap,
# and loads the tokenizers into its own anon memory. Stands in for the
# current AIPerf Kubernetes layout.
apiVersion: v1
kind: Pod
metadata:
  name: mem-containers
  labels:
    app: mem-validate
    mode: containers
spec:
  restartPolicy: Never
  # Pin to the kind worker so we compare apples to apples with forkserver pod.
  nodeSelector:
    kubernetes.io/hostname: ${KIND_NODE}
  volumes:
    - name: shared
      emptyDir: {}
    - name: hf-cache
      emptyDir:
        sizeLimit: 2Gi
  initContainers:
    # Warm the HF cache once — otherwise the N containers race to download
    # the tokenizers and only the Anon (not file-backed) part of the final
    # state actually converges. With a prewarmed cache all workers load
    # the same on-disk files identically.
    - name: warm-hf-cache
      image: localhost/aiperf-mem-probe:latest
      imagePullPolicy: Never
      command: ["/app/.venv/bin/python", "-c"]
      args:
        - |
          from transformers import AutoTokenizer
          for m in ["Qwen/Qwen3-0.6B", "openai/gpt-oss-120b"]:
              print("warming", m, flush=True)
              AutoTokenizer.from_pretrained(m, trust_remote_code=True)
      env:
        - name: HF_HOME
          value: /hf_cache
        - name: TRANSFORMERS_CACHE
          value: /hf_cache
      volumeMounts:
        - name: hf-cache
          mountPath: /hf_cache
  containers:
${WORKER_CONTAINERS}
