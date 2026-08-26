// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ML/AI code-template renderers.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;

/// A full fine-tuning training script.
pub(super) fn training_code(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let model = r.pick(MODELS)?;
    let imp = r.sample(ML_IMPORTS, 3)?;
    let (imp1, imp2, imp3) = (imp[0], imp[1], imp[2]);
    let cls = r.sample(ML_CLASSES, 2)?;
    let (cls1, cls2) = (cls[0], cls[1]);
    let m = r.sample(ML_METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(ML_VARS, 4)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let lr = ["1e-05", "2e-05", "5e-05", "0.0001", "0.0003"][r.index(5)?];
    let epochs = r.number(1, 10)?;
    let bs = [1i64, 2, 4, 8, 16, 32][r.index(6)?];
    let grad_accum = [1i64, 2, 4, 8][r.index(4)?];

    Ok(format!(
        r#"import {imp1}
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
"#
    ))
}

/// A generation/inference script.
pub(super) fn inference_code(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let model = r.pick(MODELS)?;
    let cls1 = ["AutoModelForCausalLM", "AutoModelForSeq2SeqLM"][r.index(2)?];
    let v = r.sample(ML_VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let temps: [(f64, &str); 4] = [(0.1, "0.1"), (0.3, "0.3"), (0.7, "0.7"), (1.0, "1.0")];
    let (temp_val, temp) = temps[r.index(4)?];
    let top_p = ["0.9", "0.95", "1.0"][r.index(3)?];
    let max_new = [128i64, 256, 512, 1024, 2048][r.index(5)?];
    let do_sample = if temp_val > 0.0 { "True" } else { "False" };

    Ok(format!(
        r#"import torch
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
    do_sample={do_sample},
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
"#
    ))
}

/// A fine-tuning JSON configuration carrying training, LoRA, quantization, and
/// data sections.
pub(super) fn config(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let model = r.pick(MODELS)?;
    let lr = ["1e-05", "2e-05", "5e-05", "0.0001", "0.0003"][r.index(5)?];
    let epochs = r.number(1, 10)?;
    let bs = [1i64, 2, 4, 8, 16, 32][r.index(6)?];
    let grad_accum = [1i64, 2, 4, 8][r.index(4)?];
    let max_len = [512i64, 1024, 2048, 4096][r.index(4)?];
    let warmup = ["0.03", "0.05", "0.1"][r.index(3)?];
    let lora_r = [8i64, 16, 32, 64][r.index(4)?];
    let lora_alpha = lora_r * 2;
    let quant_bits = [4i64, 8][r.index(2)?];

    Ok(format!(
        r#"{{{{
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
"#
    ))
}

/// A synthetic step, evaluation, and GPU training log.
pub(super) fn training_log(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let model_full = r.pick(MODELS)?;
    let model = model_full.rsplit('/').next().unwrap_or(model_full);
    let total_steps = r.number(500, 10000)?;
    let epoch = r.number(0, 5)?;
    let mut lines: Vec<String> = Vec::new();

    let iters = r.number(8, 15)?;
    for _ in 0..iters {
        let step = r.number(1, total_steps)?;
        let loss = r.uniform(0.3, 4.0);
        let lr_val = r.uniform(1e-6, 5e-4);
        let grad = r.uniform(0.1, 10.0);
        let tokens_per_sec = r.number(1000, 50000)?;
        let epoch_frac = epoch as f64 + step as f64 / total_steps as f64;
        let lr_str = TemplateRenderer::py_sci(lr_val, 2);
        lines.push(format!(
            "{{{{'step': {step}, 'epoch': {epoch_frac:.2}, 'loss': {loss:.4}, 'lr': {lr_str}, 'grad_norm': {grad:.3}, 'tokens_per_sec': {tokens_per_sec}}}}}"
        ));
    }

    let gpu_mem = r.uniform(10.0, 80.0);
    let gpu_util = r.number(80, 100)?;
    let eval_loss = r.uniform(0.5, 3.0);
    let eval_ppl = r.uniform(2.0, 20.0);
    let peak_extra = r.uniform(1.0, 10.0);
    let reserved_extra = r.uniform(5.0, 20.0);
    let peak_mem = gpu_mem + peak_extra;
    let reserved_mem = gpu_mem + reserved_extra;

    lines.push(format!(
        "\n[Eval] epoch={} loss={eval_loss:.4} perplexity={eval_ppl:.2}",
        epoch + 1
    ));
    lines.push(format!(
        "[GPU] memory_allocated={gpu_mem:.1}GB utilization={gpu_util}%"
    ));
    lines.push(format!(
        "[GPU] peak_memory={peak_mem:.1}GB reserved={reserved_mem:.1}GB"
    ));
    lines.push(format!(
        "[Checkpoint] Saved model checkpoint to ./checkpoints/{model}/step-{total_steps}"
    ));

    Ok(lines.join("\n") + "\n")
}

/// A CUDA traceback plus a PyTorch memory-summary table.
pub(super) fn cuda_error(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let err = r.pick(CUDA_ERRORS)?;
    let model_full = r.pick(MODELS)?;
    let model = model_full.rsplit('/').next().unwrap_or(model_full);
    let rank = r.number(0, 7)?;
    let gpu_id = r.number(0, 7)?;
    let alloc_gb = r.uniform(0.5, 16.0);
    let total_gb = [24.0f64, 40.0, 48.0, 80.0][r.index(4)?];
    let free_gb = r.uniform(0.01, 2.0);
    let _cls = r.sample(ML_CLASSES, 2)?;
    let m = r.sample(ML_METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);

    // Inline draws preserve left-to-right, top-to-bottom order.
    let line1 = r.number(50, 300)?;
    let line2 = r.number(1400, 1600)?;
    let line3 = r.number(800, 1200)?;
    let line4 = r.number(1400, 1600)?;
    let line5 = r.number(400, 800)?;
    let ooms = r.number(1, 5)?;
    let mult1 = r.number(2, 10)?;
    let mult2 = r.number(2, 10)?;

    let alloc_peak = total_gb - free_gb;
    let alloc_total = total_gb * mult1 as f64;
    let reserved_cur = total_gb - free_gb + 1.0;
    let reserved_total = total_gb * mult2 as f64;
    let total_gb_int = total_gb as i64;

    Ok(format!(
        r#"Traceback (most recent call last):
  File "train.py", line {line1}, in main
    outputs = model.{m1}(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
  File "torch/nn/modules/module.py", line {line2}, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "transformers/models/llama/modeling_llama.py", line {line3}, in {m1}
    hidden_states = self.model(input_ids, attention_mask=attention_mask)
  File "torch/nn/modules/module.py", line {line4}, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "transformers/models/llama/modeling_llama.py", line {line5}, in {m2}
    layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask)
{err}

|===========================================================================|
|                  PyTorch CUDA memory summary, device: {gpu_id}                |
|---------------------------------------------------------------------------|
|            CUDA OOMs: {ooms:>10}                                          |
|---------------------------------------------------------------------------|
|        Metric        |  Cur Usage  |  Peak Usage  |  Total Alloc  |
|---------------------------------------------------------------------------|
| Allocated memory     | {alloc_gb:>8.2} GB | {alloc_peak:>8.2} GB  | {alloc_total:>9.2} GB  |
| Reserved memory      | {reserved_cur:>8.2} GB | {total_gb:>8.2} GB  | {reserved_total:>9.2} GB  |
| Free memory          | {free_gb:>8.2} GB |              |               |
|===========================================================================|

Model: {model} | Rank: {rank} | GPU: {gpu_id} (NVIDIA A100-SXM4-{total_gb_int}GB)
"#
    ))
}
