from beartype import beartype
from huggingface_hub import login
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import os
import torch


load_dotenv()


@beartype
def generate(prompt: str, filename: str):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    login(token=os.getenv("HF_API_KEY"))

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    with torch.autocast("cuda"):
        image = pipe(
            prompt, height=384, width=384, num_inference_steps=25, guidance_scale=3.5
        ).images[0]

    image.show()
    image.save(filename)


if __name__ == "__main__":
    prompt = "A majestic steampunk boat with intricate brass and copper details sails across the open sea, its smokestacks releasing gentle plumes of steam. In the distance, the colossal figure of Cthulhu emerges ominously from the horizon, its tentacles writhing beneath a stormy, otherworldly sky. The atmosphere is eerie yet awe-inspiring, with a blend of fantasy and Lovecraftian horror."
    generate(prompt, "steampunk_boat.png")
