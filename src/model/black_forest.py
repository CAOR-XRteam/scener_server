import os
import logging

from beartype import beartype
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@beartype
def generate_image(prompt: str, filename):
    import torch

    from diffusers import FluxPipeline, StableDiffusionPipeline

    from huggingface_hub import login

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

    # pipe = FluxPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    # )
    # pipe.to(
    #     "cuda"
    # ).enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    with torch.autocast("cuda"):
        image = pipe(
            prompt, height=384, width=384, num_inference_steps=25, guidance_scale=7.5
        ).images[0]

    # with torch.autocast("cuda"):
    #     image = pipe(prompt, guidance_scale=7.5).images[0]

    image.show()
    image.save(filename)


# def generate_image(prompt, filename):
#     logger.info("Generate image...")
#     logger.info(prompt)

#     HF_API_KEY = os.getenv("HF_API_KEY")

#     client = InferenceClient(provider="fal-ai", api_key=HF_API_KEY)

#     # output is a PIL.Image object
#     image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")

#     # output is a PIL.Image object
#     image.show()
#     image.save(filename, "PNG")


if __name__ == "__main__":
    prompt = "A majestic steampunk boat with intricate brass and copper details sails across the open sea, its smokestacks releasing gentle plumes of steam. In the distance, the colossal figure of Cthulhu emerges ominously from the horizon, its tentacles writhing beneath a stormy, otherworldly sky. The atmosphere is eerie yet awe-inspiring, with a blend of fantasy and Lovecraftian horror."
    generate_image(prompt, "steampunk_boat.png")
