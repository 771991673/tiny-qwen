<p align="left">
    English | <a href="README_CN.md">中文</a>
</p>

<p align="center">
    <img src="data/chat.jpg" alt="Tiny Qwen Interactive Chat">
</p>

## ✨ Tiny Qwen

A minimal, easy-to-read PyTorch re-implementation of `Qwen3` and `Qwen2.5-VL`, supporting both text + vision as well as dense and mixture of experts.

If you find Hugging Face code verbose and challenging to interpret, this repo is for you!

Join my [Discord channel](https://discord.gg/sBNnqP9gaY) for more discussion! 

## 🦋 Quick Start

**First, install PyTorch:** Check your CUDA version with `nvidia-smi` and install from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

Then install dependencies using `uv`:

```bash
pip install uv && uv venv

# activate the environment
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows

# install dependencies
uv pip install -r requirements.txt
```

Launch the interactive chat:

```bash
python run.py
```

**Note:** `Qwen3` is text-only. Use `@path/to/image.jpg` to reference images with `Qwen2.5-VL`.

```
USER: @data/test-img-1.jpg tell me what you see in this image?
✓ Found image: data/test-img-1.jpg
ASSISTANT: The image shows a vibrant sunflower field with a close-up of a sunflower...
```

## 📝 Code Examples

**Running `Qwen2.5-VL`:**

```python
from PIL import Image
from model.model import Qwen2VL
from model.processor import Processor

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2VL.from_pretrained(repo_id=model_name, device_map="auto")
processor = Processor(repo_id=model_name, vision_config=model.config.vision_config)

context = [
    "<|im_start|>user\n<|vision_start|>",
    Image.open("data/test-img-1.jpg"),
    "<|vision_end|>What's on this image?<|im_end|>\n<|im_start|>assistant\n",
]

inputs = processor(context, device="cuda")

generator = model.generate(
    input_ids=inputs["input_ids"],
    pixels=inputs["pixels"],
    d_image=inputs["d_image"],
    max_new_tokens=64,
    stream=True,
)

for token_id in generator:
    token_text = processor.tokenizer.decode([token_id])
    print(token_text, end="", flush=True)
print()
```

## 🚀 Training Qwen3V-4B-Preview

To reproduce the training of Qwen3V-4B-Preview's projection layer:

**Step 1: Pretraining (LLaVA-595K dataset)**
```bash
PYTHONPATH=. python train/s2_1_qwen3v_pretrain.py \
    --devices 8 \
    --batch_size 8 \
    --epochs 1 \
    --grad_accum 2 \
    --max_seq_len 1024 \
    --lr 5e-4 \
    --weight_decay 0 \
    --num_workers 4 \
    --precision bf16-mixed \
    --proj_out projection-pretrained.safetensors \
    --cache_dir ./cache
```

**Step 2: Instruction Tuning (LLaVA-150K dataset)**
```bash
PYTHONPATH=. python train/s2_2_qwen3v_instruct.py \
    --devices 8 \
    --batch_size 2 \
    --epochs 3 \
    --grad_accum 8 \
    --max_seq_len 1024 \
    --lr 5e-4 \
    --weight_decay 0 \
    --num_workers 4 \
    --precision bf16-mixed \
    --proj_out projection-instruct.safetensors \
    --cache_dir ./cache \
    --pretrained_proj projection-pretrained.safetensors \
    --freeze_llm
```

**Training Notes:**
- Uses 8x H100 GPUs (can be adjusted with `--devices`)
- Only trains the projection layer (`--freeze_llm`)
- Effective batch size: 128 (pretraining), 128 (instruction tuning)
- Total training time: ~1 epoch pretraining + 3 epochs instruction tuning

**Running `Qwen3`:**

```python
from model.model import Qwen3MoE
from model.processor import Processor

model_name = "Qwen/Qwen3-4B-Instruct-2507"
model = Qwen3MoE.from_pretrained(repo_id=model_name)
processor = Processor(repo_id=model_name)

context = [
    "<|im_start|>user\n<|vision_start|>",
    "<|vision_end|>Explain reverse linked list<|im_end|>\n<|im_start|>assistant\n",
]
inputs = processor(context, device="cuda")
generator = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=64,
    stream=True
)

for token_id in generator:
    token_text = processor.tokenizer.decode([token_id])
    print(token_text, end="", flush=True)
print()
```