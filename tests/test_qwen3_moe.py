import torch
from tqdm import tqdm
from model.qwen3 import Qwen3MoE
from model.processor import Processor


def test_qwen3_moe():
    model_id = "Qwen/Qwen3-4B"
    max_new_tokens = 64

    model = Qwen3MoE.from_pretrained(repo_id=model_id, device_map="auto")
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
    correct_token_ids = [151667, 198, 32313, 11, 279, 1196, 374, 10161, 911, 279, 7290, 315, 2272, 13, 2938, 594, 264, 2409, 3405, 13, 358, 1184, 311, 5486, 419, 15516, 13, 5512, 11, 358, 1265, 24645, 429, 432, 594, 264, 40803, 3405, 448, 902, 3175, 4226, 13, 33396, 1251, 323, 26735, 614, 2155, 38455, 382, 40, 1265, 6286, 429, 432, 594, 264, 8544, 35031, 553, 60687, 11, 13923]

    correct_response = """<think>
Okay, the user is asking about the meaning of life. That's a big question. I need to approach this carefully. First, I should acknowledge that it's a philosophical question with no single answer. Different people and cultures have different perspectives.

I should mention that it's a topic explored by philosophers, scientists"""
    # fmt: on

    assert token_ids == correct_token_ids, "Token IDs do not match"
    assert response == correct_response, "Response does not match"

    print("Test passed")
