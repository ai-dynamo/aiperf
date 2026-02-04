# SGLang Image Generation

## Overview
This guide shows how to benchmark image generation APIs using SGLang and AIPerf. You'll learn how to set up the server, create an input file, and run the benchmark. You'll also learn how to view the results, and even extract the generated images!

## References
For the most up-to-date information, please refer to the following resources:
- [OpenAI Image Generation API](https://platform.openai.com/docs/api-reference/images/create)
- [SGLang Image Generation Installation Guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/install.md)
- [SGLang Image Generation CLI](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md)

## Setting up the server

**Login to Hugging Face, and accept the terms of use for the following model:**
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

**Export your Hugging Face token as an environment variable:**
```bash
export HF_TOKEN=<your-huggingface-token>
```

**Start the SGLang Docker container:**
```bash
docker run --gpus all \
    --shm-size 32g \
    -it \
    --rm \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:dev
```

> [!NOTE]
> The following steps are to be performed _inside_ the SGLang Docker container.

**Install the dependencies:**
```bash
uv pip install yunchang remote_pdb imageio diffusers diffusion --system
```

**Set the server arguments:**
> [!IMPORTANT]
> The following arguments will setup the SGLang server to use the FLUX.1-dev model on a single GPU, on port 30000.
> You can modify these arguments to use a different model, different number of GPUs, different port, etc.
> See the [SGLang Image Generation CLI](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md) for more details.
```bash
SERVER_ARGS=(   --model-path black-forest-labs/FLUX.1-dev   --text-encoder-cpu-offload   --pin-cpu-memory   --num-gpus 1   --port 30000 --host 0.0.0.0 )
```

**Start the SGLang server:**
```bash
sglang serve "${SERVER_ARGS[@]}"
```

**Wait until the server is ready** (watch the logs for the following message):
```bash
Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

## Running the benchmark (basic usage)

> [!NOTE]
> The following steps are to be performed on your local machine. (_outside_ the SGLang Docker container.)


### Text-to-Image Generation Using Input File
**Create an input file:**

```bash
cat > image_prompts.jsonl << 'EOF'
{"text": "A serene mountain landscape at sunset"}
{"text": "A futuristic city with flying cars"}
{"text": "A cute robot playing with a kitten"}
EOF
```

**Run the benchmark:**
```bash
aiperf profile \
  --model black-forest-labs/FLUX.1-dev \
  --tokenizer gpt2 \
  --url http://localhost:30000 \
  --endpoint-type image_generation \
  --input-file image_prompts.jsonl \
  --custom-dataset-type single_turn \
  --extra-inputs size:512x512 \
  --extra-inputs quality:standard \
  --concurrency 1 \
  --request-count 3
```

**Done!** This sends 3 requests to `http://localhost:30000/v1/images/generations`

**Sample Output (Successful Run):**
```
                                       NVIDIA AIPerf | Image Generation Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃                            Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│              Request Latency (ms) │ 12,617.58 │ 12,251.41 │ 12,954.04 │ 12,947.91 │ 12,892.69 │ 12,647.29 │ 287.62 │
│    Input Sequence Length (tokens) │      6.67 │      6.00 │      7.00 │      7.00 │      7.00 │      7.00 │   0.47 │
│ Request Throughput (requests/sec) │      0.08 │         - │         - │         - │         - │         - │      - │
│          Request Count (requests) │      3.00 │         - │         - │         - │         - │         - │      - │
└───────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────┘
```


### Text-to-Image Generation Using Synthetic Inputs
```bash
aiperf profile \
  --model black-forest-labs/FLUX.1-dev \
  --tokenizer gpt2 \
  --url http://localhost:30000 \
  --endpoint-type image_generation \
  --extra-inputs size:512x512 \
  --extra-inputs quality:standard \
  --synthetic-input-tokens-mean 150 \
  --synthetic-input-tokens-stddev 30 \
  --concurrency 1 \
  --request-count 3
```

**Done!** This sends 3 requests to `http://localhost:30000/v1/images/generations`

**Sample Output (Successful Run):**
```
                                       NVIDIA AIPerf | Image Generation Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃                            Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│              Request Latency (ms) │ 12,173.18 │ 11,918.37 │ 12,503.38 │ 12,495.27 │ 12,422.26 │ 12,097.79 │ 244.71 │
│    Input Sequence Length (tokens) │    137.00 │    107.00 │    153.00 │    152.96 │    152.60 │    151.00 │  21.23 │
│ Request Throughput (requests/sec) │      0.08 │         - │         - │         - │         - │         - │      - │
│          Request Count (requests) │      3.00 │         - │         - │         - │         - │         - │      - │
└───────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────┘
```


## Running the benchmark (advanced usage)

**Create an input file:**
```bash
cat > image_prompts.jsonl << 'EOF'
{"text": "A serene mountain landscape at sunset"}
{"text": "A futuristic city with flying cars"}
{"text": "A cute robot playing with a kitten"}
EOF
```

**Run the benchmark:**

>[!IMPORTANT]
> Use `--export-level raw` to get the raw input/output payloads.

```bash
aiperf profile \
  --model black-forest-labs/FLUX.1-dev \
  --tokenizer gpt2 \
  --url http://localhost:30000 \
  --endpoint-type image_generation \
  --input-file image_prompts.jsonl \
  --custom-dataset-type single_turn \
  --extra-inputs size:512x512 \
  --extra-inputs quality:standard \
  --concurrency 1 \
  --request-count 3 \
  --export-level raw
```

### Viewing the generated images

**Extract the generated images:**<br>
Copy the following code into a file called `extract_images.py`:
```python extract_images.py
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract base64-encoded images from AIPerf JSONL output file."""
import base64
import json
import os
from pathlib import Path
import sys

# Read input file path
input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('artifacts/black-forest-labs_FLUX.1-dev-openai-image_generation-concurrency1/profile_export_raw.jsonl')
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('extracted_images')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process each line in the JSONL file
with open(input_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        record = json.loads(line)

        # Extract images from responses
        for response in record.get('responses', []):
            response_data = json.loads(response.get('text', '{}'))

            for data_idx, item in enumerate(response_data.get('data', [])):
                if b64_image := item.get('b64_json'):
                    # Decode and save image
                    image_data = base64.b64decode(b64_image)
                    filename = output_dir / f"image_{line_num:04d}_{data_idx:02d}.jpg"

                    with open(filename, 'wb') as img_file:
                        img_file.write(image_data)

                    print(f"Extracted: {filename.resolve()}")
```

**Run the script:**
> [!TIP]
> The script is setup to use the default directory and file names for the input and output files, but can be modified to use different files.<br>
>
> Usage: `python extract_images.py <input_file> <output_dir>`

```bash
python extract_images.py
```
**Output:**
```
Extracted: /path/to/extracted_images/image_0001_00.jpg
Extracted: /path/to/extracted_images/image_0001_01.jpg
Extracted: /path/to/extracted_images/image_0001_02.jpg
```

**View the generated images:**

Prompt:
```
{"text": "A serene mountain landscape at sunset"}
```
![](../media/extracted_images/image_0001_00_00.jpg)

Prompt:
```
{"text": "A futuristic city with flying cars"}
```
![](../media/extracted_images/image_0002_00_00.jpg)

Prompt:
```
{"text": "A cute robot playing with a kitten"}
```
![](../media/extracted_images/image_0003_00_00.jpg)

## Conclusion

You've successfully set up SGLang, run your first image generation benchmarks, and learned how to extract and view the generated images. You can now experiment with different models, prompts, and concurrency settings to optimize your image generation workloads.

Now go forth and generate!
