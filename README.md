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

I recommend using `uv` and creating a virtual environment:

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
from model.qwen2_5_vl import Qwen2VL
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

**Running `Qwen3`:**

```python
from model.qwen3 import Qwen3MoE
from model.processor import Processor

model_name = "Qwen/Qwen3-4B-Instruct-2507"
model = Qwen3MoE.from_pretrained(repo_id=model_name)
processor = Processor(repo_id=model_name)

context = [
    "<|im_start|>user\n",
    "Explain reverse linked list<|im_end|>\n<|im_start|>assistant\n",
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