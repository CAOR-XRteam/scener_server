import io
import os
import torch

from beartype import beartype
from diffusers import FluxPipeline
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image


load_dotenv()


@beartype
def convert_image_to_bytes(image_path: str):
    try:
        with Image.open(image_path) as image:
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            return byte_arr.getvalue()
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        raise


@beartype
def generate(prompt: str, filename: str):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    with torch.autocast("cuda"):
        image = pipe(
            prompt, height=384, width=384, num_inference_steps=25, guidance_scale=3.5
        ).images[0]

    image.show()
    image.save(filename)

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    prompt = "A majestic steampunk boat with intricate brass and copper details sails across the open sea, its smokestacks releasing gentle plumes of steam. In the distance, the colossal figure of Cthulhu emerges ominously from the horizon, its tentacles writhing beneath a stormy, otherworldly sky. The atmosphere is eerie yet awe-inspiring, with a blend of fantasy and Lovecraftian horror."
    generate(prompt, "steampunk_boat.png")
