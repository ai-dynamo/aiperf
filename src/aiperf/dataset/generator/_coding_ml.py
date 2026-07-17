# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ML/CUDA template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CUDA_ERRORS,
    _ML_CLASSES,
    _ML_IMPORTS,
    _ML_METHODS,
    _ML_VARS,
    _MODEL_NAMES,
)


class _MlMixin:
    def _gen_ml_training_code(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES)
        imp1, imp2, imp3 = r.sample(list(_ML_IMPORTS), 3)
        cls1, cls2 = r.sample(list(_ML_CLASSES), 2)
        m1, m2 = r.sample(list(_ML_METHODS), 2)
        v1, v2, v3, v4 = r.sample(list(_ML_VARS), 4)
        lr = r.choice([1e-5, 2e-5, 5e-5, 1e-4, 3e-4])
        epochs = r.randint(1, 10)
        bs = r.choice([1, 2, 4, 8, 16, 32])
        grad_accum = r.choice([1, 2, 4, 8])

        return f"""\
import {imp1}
import {imp2}
from {imp3} import {cls1}, {cls2}

model_name = "{model}"
tokenizer = {cls2}.from_pretrained(model_name)
model = {cls1}.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

train_dataset = datasets.load_dataset("json", data_files="train.jsonl", split="train")

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs={epochs},
    per_device_train_batch_size={bs},
    gradient_accumulation_steps={grad_accum},
    learning_rate={lr},
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
)

optimizer = torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay=0.01)

for epoch in range({epochs}):
    model.train()
    for step, batch in enumerate(train_loader):
        {v1} = batch["{v1}"].to("cuda")
        {v2} = batch["{v2}"].to("cuda")
        outputs = model({m1}={v1}, {v2}={v2})
        {v3} = outputs.{v3}
        {v3}.{m2}()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {{epoch}} Step {{step}} {v3}: {{{{{v3}.item():.4f}}}}")

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
"""

    def _gen_ml_inference_code(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES)
        cls1 = r.choice(("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"))
        v1, v2, v3 = r.sample(list(_ML_VARS), 3)
        temp = r.choice([0.1, 0.3, 0.7, 1.0])
        top_p = r.choice([0.9, 0.95, 1.0])
        max_new = r.choice([128, 256, 512, 1024, 2048])

        return f"""\
import torch
from transformers import {cls1}, AutoTokenizer, GenerationConfig

model_name = "{model}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = {cls1}.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

generation_config = GenerationConfig(
    max_new_tokens={max_new},
    temperature={temp},
    top_p={top_p},
    do_sample={"True" if temp > 0 else "False"},
    repetition_penalty=1.1,
)

prompt = "Explain the architecture of a transformer model."
{v1} = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    {v2} = model.generate(
        **{v1},
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
    )

{v3} = tokenizer.batch_decode({v2}[:, {v1}["{v1}"].shape[-1]:], skip_special_tokens=True)
print({v3}[0])
"""

    def _gen_ml_config(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES)
        lr = r.choice([1e-5, 2e-5, 5e-5, 1e-4, 3e-4])
        epochs = r.randint(1, 10)
        bs = r.choice([1, 2, 4, 8, 16, 32])
        grad_accum = r.choice([1, 2, 4, 8])
        max_len = r.choice([512, 1024, 2048, 4096])
        warmup = r.choice([0.03, 0.05, 0.1])
        lora_r = r.choice([8, 16, 32, 64])
        lora_alpha = lora_r * 2
        quant_bits = r.choice([4, 8])

        return f"""\
{{{{
  "model_name_or_path": "{model}",
  "torch_dtype": "bfloat16",
  "attn_implementation": "flash_attention_2",
  "max_seq_length": {max_len},
  "training": {{{{
    "num_train_epochs": {epochs},
    "per_device_train_batch_size": {bs},
    "gradient_accumulation_steps": {grad_accum},
    "learning_rate": {lr},
    "weight_decay": 0.01,
    "warmup_ratio": {warmup},
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "bf16": true,
    "gradient_checkpointing": true,
    "optim": "adamw_torch_fused"
  }}}},
  "lora": {{{{
    "r": {lora_r},
    "lora_alpha": {lora_alpha},
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
  }}}},
  "quantization": {{{{
    "load_in_{quant_bits}bit": true,
    "bnb_{quant_bits}bit_compute_dtype": "bfloat16",
    "bnb_{quant_bits}bit_quant_type": "nf4",
    "bnb_{quant_bits}bit_use_double_quant": true
  }}}},
  "data": {{{{
    "dataset_name": "train.jsonl",
    "max_length": {max_len},
    "packing": true,
    "num_proc": 8
  }}}}
}}}}
"""

    def _gen_ml_training_log(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES).split("/")[-1]
        total_steps = r.randint(500, 10000)
        epoch = r.randint(0, 5)
        lines = []

        for _ in range(r.randint(8, 15)):
            step = r.randint(1, total_steps)
            loss = r.uniform(0.3, 4.0)
            lr_val = r.uniform(1e-6, 5e-4)
            grad = r.uniform(0.1, 10.0)
            tokens_per_sec = r.randint(1000, 50000)
            lines.append(
                f"{{{{'step': {step}, 'epoch': {epoch + step / total_steps:.2f}, "
                f"'loss': {loss:.4f}, 'lr': {lr_val:.2e}, "
                f"'grad_norm': {grad:.3f}, 'tokens_per_sec': {tokens_per_sec}}}}}"
            )

        gpu_mem = r.uniform(10, 80)
        gpu_util = r.randint(80, 100)
        eval_loss = r.uniform(0.5, 3.0)
        eval_ppl = r.uniform(2.0, 20.0)

        lines.append(
            f"\n[Eval] epoch={epoch + 1} loss={eval_loss:.4f} perplexity={eval_ppl:.2f}"
        )
        lines.append(f"[GPU] memory_allocated={gpu_mem:.1f}GB utilization={gpu_util}%")
        lines.append(
            f"[GPU] peak_memory={gpu_mem + r.uniform(1, 10):.1f}GB "
            f"reserved={gpu_mem + r.uniform(5, 20):.1f}GB"
        )
        lines.append(
            f"[Checkpoint] Saved model checkpoint to ./checkpoints/{model}/step-{total_steps}"
        )

        return "\n".join(lines) + "\n"

    def _gen_cuda_error(self) -> str:
        r = self._template_rng
        err = r.choice(_CUDA_ERRORS)
        model = r.choice(_MODEL_NAMES).split("/")[-1]
        rank = r.randint(0, 7)
        gpu_id = r.randint(0, 7)
        alloc_gb = r.uniform(0.5, 16.0)
        total_gb = r.choice([24.0, 40.0, 48.0, 80.0])
        free_gb = r.uniform(0.01, 2.0)
        cls1, cls2 = r.sample(list(_ML_CLASSES), 2)
        m1, m2 = r.sample(list(_ML_METHODS), 2)

        return f"""\
Traceback (most recent call last):
  File "train.py", line {r.randint(50, 300)}, in main
    outputs = model.{m1}(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
  File "torch/nn/modules/module.py", line {r.randint(1400, 1600)}, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "transformers/models/llama/modeling_llama.py", line {r.randint(800, 1200)}, in {m1}
    hidden_states = self.model(input_ids, attention_mask=attention_mask)
  File "torch/nn/modules/module.py", line {r.randint(1400, 1600)}, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "transformers/models/llama/modeling_llama.py", line {r.randint(400, 800)}, in {m2}
    layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask)
{err}

|===========================================================================|
|                  PyTorch CUDA memory summary, device: {gpu_id}                |
|---------------------------------------------------------------------------|
|            CUDA OOMs: {r.randint(1, 5):>10}                                          |
|---------------------------------------------------------------------------|
|        Metric        |  Cur Usage  |  Peak Usage  |  Total Alloc  |
|---------------------------------------------------------------------------|
| Allocated memory     | {alloc_gb:>8.2f} GB | {total_gb - free_gb:>8.2f} GB  | {total_gb * r.randint(2, 10):>9.2f} GB  |
| Reserved memory      | {total_gb - free_gb + 1:>8.2f} GB | {total_gb:>8.2f} GB  | {total_gb * r.randint(2, 10):>9.2f} GB  |
| Free memory          | {free_gb:>8.2f} GB |              |               |
|===========================================================================|

Model: {model} | Rank: {rank} | GPU: {gpu_id} (NVIDIA A100-SXM4-{int(total_gb)}GB)
"""
