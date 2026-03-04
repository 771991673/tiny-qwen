import os
import re
import torch
import traceback

import typer
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
import questionary
from questionary import Choice, Style
from rich.console import Console
from rich.text import Text

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

disable_progress_bars()

from model.processor import Processor
from model.model import Qwen3_5

ASCII_LOGO = """
██╗    ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██╗    ██╗███████╗███╗   ██╗
╚██╗   ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ╚██╗     ██║   ██║██╔██╗ ██║ ╚████╔╝    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██╔╝     ██║   ██║██║╚██╗██║  ╚██╔╝     ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
██╔╝      ██║   ██║██║ ╚████║   ██║      ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
╚═╝       ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""

STARTING_HELP_TEXT = """
Welcome to Tiny-Qwen Interactive Chat!

Tips:
1. /help for more information.
2. /exit or Ctrl+C to exit.
"""


HELP_TEXT = """
Available commands:
/help - Show this help message
/exit - Exit the application

Use @relative/path/to/image.jpg to include images in your messages.
"""

ALL_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]

STYLE = Style(
    [
        ("question", "bold"),
        ("selected", "fg:#000000 bg:#face0a bold"),
        ("highlighted", "fg:#face0a bold"),
        ("instruction", "fg:#888888"),
        ("separator", "fg:#666666"),
        ("text", ""),
        ("qmark", "fg:#face0a"),
    ]
)

console = Console(highlight=False)
app = typer.Typer(add_completion=False)


def parse_user_input(text):
    image_pattern = r"@([^\s]+\.(?:jpg|jpeg|png|gif|webp))"
    matches = list(re.finditer(image_pattern, text, re.IGNORECASE))

    if not matches:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    content = []
    last_end = 0

    for match in matches:
        if match.start() > last_end:
            text_part = text[last_end : match.start()].strip()
            if text_part:
                content.append({"type": "text", "text": text_part})

        image_path = match.group(1)
        if os.path.exists(image_path):
            content.append({"type": "image", "image": image_path})
        else:
            console.print(f"Warning: Image not found: {image_path}", style="yellow")

        last_end = match.end()

    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            content.append({"type": "text", "text": remaining_text})

    return [{"role": "user", "content": content}]


def generate_local_response(messages, model, processor, max_tokens=2048):
    device = next(model.parameters()).device
    inputs = processor(messages, add_generation_prompt=True, device=device)

    stop_tokens = [151645, 151644, 151643]  # <|im_end|>, <|im_start|>, <|endoftext|>
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": max_tokens,
        "stop_tokens": stop_tokens,
    }

    if inputs["pixels"] is not None:
        generation_kwargs["pixels"] = inputs["pixels"]
    if inputs["d_image"] is not None:
        generation_kwargs["d_image"] = inputs["d_image"]

    token_generator = model.generate_stream(**generation_kwargs)
    generated_tokens = []
    previous_text = ""
    for token_id in token_generator:
        generated_tokens.append(token_id)
        current_text = processor.tokenizer.decode(generated_tokens)
        new_text = current_text[len(previous_text) :]
        if new_text:
            previous_text = current_text
            yield new_text


@app.command()
def main():
    try:
        # clear terminal
        os.system("cls" if os.name == "nt" else "clear")

        # show logo and help message
        yellow_logo = Text(ASCII_LOGO, style="#face0a")
        console.print(yellow_logo)
        console.print(STARTING_HELP_TEXT)

        # select model variant
        selected_model_variant = questionary.select(
            message="Select model variant",
            choices=[Choice(variant, variant) for variant in ALL_MODELS],
            pointer=">",
            qmark="",
            style=STYLE,
        ).ask()

        if not selected_model_variant:
            return

        # load model
        hf_repo_id = selected_model_variant
        with console.status(f"[bold #face0a]Loading {hf_repo_id}...", spinner="dots"):
            try:
                weights_path = snapshot_download(repo_id=hf_repo_id, cache_dir=".cache")
                processor = Processor.from_pretrained(hf_repo_id)
                model = Qwen3_5.from_pretrained(
                    weights_path=weights_path, device_map="auto"
                )
                model.eval()
                model = torch.compile(model)
            except Exception as e:
                console.print(f"Failed to load model: {e}")
                return

        if model is None or processor is None:
            console.print("Failed to initialize processor. Exiting...", style="red")
            return

        # start REPL
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "The assistant anwsers user's question concisely and accurately, ideally in a paragraph or two if not a few sentences. Since the assistant is interacting with the user in a CLI, it will only respond in plain text, avoiding emoji or markdown unless specifically requested.",
                    }
                ],
            }
        ]
        while True:
            user_input = input("\nUSER: ").strip()

            if user_input == "/exit":
                console.print("Goodbye!")
                break
            elif user_input == "/help":
                console.print(HELP_TEXT)
                continue
            elif not user_input:
                continue

            current_messages = parse_user_input(user_input)
            messages.extend(current_messages)

            try:
                response_segments = []
                print("QWEN: ", end="", flush=True)
                for segment in generate_local_response(messages, model, processor):
                    print(segment, end="", flush=True)
                    response_segments.append(segment)
                response = "".join(response_segments)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    }
                )
                print()
            except Exception as e:
                console.print(f"Error generating response: {e}", style="red")
                console.print(traceback.format_exc(), style="red")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        raise


if __name__ == "__main__":
    app()
