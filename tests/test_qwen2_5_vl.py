import torch
from PIL import Image
from tqdm import tqdm
from model.qwen2_5_vl import Qwen2VL
from model.processor import Processor


def test_qwen2_5_vl():

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens = 64
    model = Qwen2VL.from_pretrained(repo_id=model_id, device_map="auto")
    processor = Processor(repo_id=model_id, vision_config=model.config.vision_config)

    model = torch.compile(model)

    context = [
        "<|im_start|>user\n<|vision_start|>",
        Image.open("data/test-img-3.jpeg"),
        "<|vision_end|>whats funny about this image?<|im_end|>\n<|im_start|>assistant\n",
    ]

    inputs = processor(context, device="cuda")

    # Stream tokens and collect them
    token_ids = []
    response_tokens = []
    for token_id in tqdm(
        model.generate(
            input_ids=inputs["input_ids"],
            pixels=inputs["pixels"],
            d_image=inputs["d_image"],
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
    correct_token_ids = [785, 2168, 7952, 311, 387, 264, 69846, 323, 64523, 72664, 315, 264, 6109, 1380, 264, 1874, 315, 9898, 11, 10767, 69144, 476, 4428, 19970, 11, 525, 11259, 389, 862, 47319, 14201, 304, 264, 89773, 4573, 13, 576, 9898, 525, 27802, 304, 264, 1616, 429, 3643, 1105, 1401, 1075, 807, 525, 11435, 389, 862, 6078, 11, 892, 374, 458, 18511, 323, 469, 938, 13929, 13]

    correct_response = "The image appears to be a humorous and surreal depiction of a scene where a group of animals, possibly rabbits or similar creatures, are standing on their hind legs in a snowy environment. The animals are arranged in a way that makes them look like they are walking on their hands, which is an unusual and comical sight."
    # fmt: on

    assert token_ids == correct_token_ids, "Token IDs do not match"
    assert response == correct_response, "Response does not match"

    print("Test passed")
