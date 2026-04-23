# Forkserver-mode pod: one container runs the parent; parent os.forks() N
# children that share its anon import heap + tokenizer state via CoW.
# Stands in for a hypothetical "workers-as-subprocesses" AIPerf layout
# with tokenizers lifted into the forkserver preload.
apiVersion: v1
kind: Pod
metadata:
  name: mem-forkserver
  labels:
    app: mem-validate
    mode: forkserver
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: ${KIND_NODE}
  volumes:
    - name: shared
      emptyDir: {}
    - name: hf-cache
      emptyDir:
        sizeLimit: 2Gi
  initContainers:
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
    - name: forkserver
      image: localhost/aiperf-mem-probe:latest
      imagePullPolicy: Never
      args:
        - forkserver
        - --prefix
        - fs
        - --n
        - "${N_CHILDREN}"
        - --tokenizers
        - Qwen/Qwen3-0.6B
        - openai/gpt-oss-120b
      env:
        - name: HF_HOME
          value: /hf_cache
        - name: TRANSFORMERS_CACHE
          value: /hf_cache
        - name: PYTHONUNBUFFERED
          value: "1"
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
      volumeMounts:
        - name: shared
          mountPath: /shared
        - name: hf-cache
          mountPath: /hf_cache
