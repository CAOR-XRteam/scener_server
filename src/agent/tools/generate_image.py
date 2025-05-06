from model import black_forest
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from beartype import beartype
from loguru import logger
from colorama import Fore
from pathlib import Path


class GenerateImageToolInput(BaseModel):
    scene_json: dict = Field(
        description="The JSON representing the decomposed scene, confirmed by the user."
    )


@tool(args_schema=GenerateImageToolInput)
def generate_image(scene_json: dict):
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""

    logger.info(f"Using tool {Fore.GREEN}{'image_generation'}{Fore.RESET}")
    logger.info(f"\nAgent: Decomposed JSON received: {scene_json}. Generating image...")

    # Retrieve list of to-generated objects
    try:
        objects_to_generate = scene_json.get("scene", {}).get(
            "objects", []
        )
        logger.info(f"Agent: Decomposed objects to generate: {objects_to_generate}")
    except Exception as e:
        logger.error(f"Failed to extract objects from JSON: {e}")
        return f"[Error during image generation: {e}]"

    if not objects_to_generate:
        logger.info("Agent: The decomposition resulted in no specific objects to generate images for.")
        return "[No objects to generate images for.]"

    # Objects generation
    for i, obj in enumerate(objects_to_generate):
        if isinstance(obj, dict) and obj.get("prompt"):
            logger.info(f"Agent: Generating image for object {i+1}: {obj['prompt']}")
            obj_name = obj.get("name", f"object_{i+1}").replace(" ", "_").lower()
            dir_path = str(Path(__file__).resolve().parents[3] / "media" / "temp")
            filename = dir_path + "/" + obj_name + ".png"
            black_forest.generate(obj["prompt"], filename)
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
                    "prompt": "A plush, cream-colored couch with a low back and rolled arms, front camera view, placed on a white and empty background, completely detached from its surroundings."
                },
                {
                    "name": "gray_cat",
                    "type": "prop",
                    "material": "glossy_fur",
                    "prompt": "A sleek, gray cat with bright green eyes, front camera view, placed on a white and empty background, completely detached from its surroundings."
                },
                {
                    "name": "living_room",
                    "type": "room",
                    "material": "polished_wood",
                    "prompt": "A squared room, room view from the outside with a distant 3/4 top-down perspective, placed on a white and empty background, completely detached from its surroundings."
                }
            ]
        }
    }


    res = generate_image.invoke({"scene_json": scene_dict})  # ✅ Pass a dict
    print(res)
