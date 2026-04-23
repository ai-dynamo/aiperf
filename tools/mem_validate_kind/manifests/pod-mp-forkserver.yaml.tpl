# mp-forkserver mode: real multiprocessing.set_forkserver_preload with a
# side-effect module that instantiates the tokenizers inside the forkserver
# helper. Children spawned via the forkserver context CoW-share the
# tokenizer pages.
apiVersion: v1
kind: Pod
metadata:
  name: mem-mp-forkserver
  labels:
    app: mem-validate
    mode: mp-forkserver
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
        - mp-forkserver
        - --prefix
        - mp
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
        # Verifies the tokenizer_preload module honors env var config. This
        # same pattern would apply in real AIPerf: operator injects model
        # IDs, forkserver preload instantiates them once per pod.
        - name: AIPERF_PRELOAD_TOKENIZERS
          value: "Qwen/Qwen3-0.6B,openai/gpt-oss-120b"
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
      volumeMounts:
        - name: shared
          mountPath: /shared
        - name: hf-cache
          mountPath: /hf_cache
