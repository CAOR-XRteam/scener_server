from colorama import Fore
from langchain_core.tools import tool
from lib import logger
from model import black_forest
import os
from pathlib import Path
from pydantic import BaseModel, Field


class GenerateImageToolInput(BaseModel):
    improved_decomposed_input: dict = Field(
        description="The JSON representing the decomposed scene."
    )


@tool(args_schema=GenerateImageToolInput)
def generate_image(improved_decomposed_input: dict):
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""

    logger.info(f"Using tool {Fore.GREEN}{'image_generation'}{Fore.RESET}")
    logger.info(
        f"\nAgent: Decomposed JSON received: {improved_decomposed_input}. Generating image..."
    )

    # Retrieve list of to-generated objects
    try:
        objects_to_generate = improved_decomposed_input.get("scene", {}).get(
            "objects", []
        )
        logger.info(f"Agent: Decomposed objects to generate: {objects_to_generate}")
    except Exception as e:
        logger.error(f"Failed to extract objects from JSON: {e}")
        return f"[Error during image generation: {e}]"

    if not objects_to_generate:
        logger.info(
            "Agent: The decomposition resulted in no specific objects to generate images for."
        )
        return "[No objects to generate images for.]"

    # Objects generation
    for i, obj in enumerate(objects_to_generate):
        if isinstance(obj, dict) and obj.get("prompt"):
            logger.info(f"Agent: Generating image for object {i+1}: {obj['prompt']}")
            obj_name = obj.get("name", f"object_{i+1}").replace(" ", "_").lower()
            dir_path = str(Path(__file__).resolve().parents[3] / "media" / "temp")
            os.makedirs(dir_path, exist_ok=True)
            filename = dir_path + "/" + obj_name + ".png"
            try:
                black_forest.generate(obj["prompt"], filename)
            except Exception as e:
                logger.error(f"Failed to generate image:{e}")
                pass
        else:
            logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
            logger.info(f"\n[Skipping object {i+1} - missing prompt]")

    logger.info("\nAgent: Image generation process complete.")
    return f"Image generation process complete."


if __name__ == "__main__":
    scene_dict = {
        "scene": {
            "objects": [
                {
                    "name": "cream_couch",
                    "type": "furniture",
                    "material": "plush_fabric",
                    "prompt": "A plush, cream-colored couch with a low back and rolled arms, front camera view, placed on a white and empty background, completely detached from its surroundings.",
                },
                {
                    "name": "gray_cat",
                    "type": "prop",
                    "material": "glossy_fur",
                    "prompt": "A sleek, gray cat with bright green eyes, front camera view, placed on a white and empty background, completely detached from its surroundings.",
                },
                {
                    "name": "living_room",
                    "type": "room",
                    "material": "polished_wood",
                    "prompt": "A squared room, room view from the outside with a distant 3/4 top-down perspective, placed on a white and empty background, completely detached from its surroundings.",
                },
            ]
        }
    }

    res = generate_image.invoke(
        {"improved_decomposed_input": scene_dict}
    )  # ✅ Pass a dict
    print(res)
