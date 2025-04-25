import os
import logging

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def generate_image(prompt):
    logger.info("Generate image...")
    logger.info(prompt)

    HF_API_KEY = os.getenv("HF_API_KEY")

    client = InferenceClient(provider="fal-ai", api_key=HF_API_KEY)

    # output is a PIL.Image object
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")

    # output is a PIL.Image object
    image.show()
    image.save("output_image.jpg")


if __name__ == "__main__":
    prompt = "A majestic steampunk boat with intricate brass and copper details sails across the open sea, its smokestacks releasing gentle plumes of steam. In the distance, the colossal figure of Cthulhu emerges ominously from the horizon, its tentacles writhing beneath a stormy, otherworldly sky. The atmosphere is eerie yet awe-inspiring, with a blend of fantasy and Lovecraftian horror."
    generate_image(prompt)
