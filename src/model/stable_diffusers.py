import torch

from beartype import beartype
from diffusers import StableDiffusion3Pipeline, EulerDiscreteScheduler


@beartype
def generate(prompt: str, filename: str):
    model_id = "stabilityai/stable-diffusion-3.5-medium"

    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]

    image.save(filename)
    image.show()

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    prompt = "A majestic steampunk boat with intricate brass and copper details sails across the open sea, its smokestacks releasing gentle plumes of steam. In the distance, the colossal figure of Cthulhu emerges ominously from the horizon, its tentacles writhing beneath a stormy, otherworldly sky. The atmosphere is eerie yet awe-inspiring, with a blend of fantasy and Lovecraftian horror."
    generate(prompt, "steampunk_boat.png")
