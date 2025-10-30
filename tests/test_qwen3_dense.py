import torch
from tqdm import tqdm
from model.qwen3 import Qwen3Dense
from model.processor import Processor


def test_qwen3_dense():
    model_id = "Qwen/Qwen3-1.7B"
    max_new_tokens = 64

    model = Qwen3Dense.from_pretrained(repo_id=model_id, device_map="auto")
    model = torch.compile(model)

    processor = Processor(repo_id=model_id)

    context = [
        "<|im_start|>user\nwhat is the meaning of life?<|im_end|>\n<|im_start|>assistant\n",
    ]

    input_ids = processor(context)["input_ids"].to("cuda")

    # Stream tokens and collect them
    token_ids = []
    response_tokens = []
    for token_id in tqdm(
        model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stream=True,
        ),
        total=max_new_tokens,
        desc="Generating",
        unit="tok",
    ):
        token_text = processor.tokenizer.decode([token_id])
        token_ids.append(token_id)
        response_tokens.append(token_text)

    response = "".join(response_tokens)

    # fmt: off
    correct_token_ids = [151667, 198, 32313, 11, 279, 1196, 374, 10161, 11, 330, 3838, 374, 279, 7290, 315, 2272, 7521, 2938, 594, 264, 11416, 3405, 11, 714, 432, 594, 1083, 5020, 7205, 13, 358, 1184, 311, 1281, 2704, 358, 2621, 432, 2041, 1660, 2238, 39046, 13, 6771, 752, 1191, 553, 60608, 429, 279, 3405, 374, 27155, 323, 702, 1012, 4588, 553, 1657, 1251, 6814, 3840, 382, 5338]

    correct_response = """<think>
Okay, the user is asking, "What is the meaning of life?" That's a classic question, but it's also pretty broad. I need to make sure I address it without being too vague. Let me start by acknowledging that the question is profound and has been asked by many people throughout history.

First"""
    # fmt: on

    assert token_ids == correct_token_ids, "Token IDs do not match"
    assert response == correct_response, "Response does not match"

    print("Test passed")
