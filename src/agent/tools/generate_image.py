from model.black_forest import generate_image
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from beartype import beartype
from loguru import logger
from colorama import Fore


class GenerateImageToolInput(BaseModel):
    decomposed_user_input: dict = Field(
        description="The JSON representing the decomposed scene, confirmed by the user."
    )


@tool(args_schema=GenerateImageToolInput)
def generate_image(self, decomposed_user_input: dict):
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""

    logging.info(f"Agent: Received decomposed user input: {decomposed_user_input}")
    logger.info(f"\nAgent: Decomposed JSON received: {decomposed_user_input}. Generating image...")
    
    try:
        objects_to_generate = decomposed_user_input.get("scene", {}).get(
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

    for i, obj in enumerate(objects_to_generate):
        if isinstance(obj, dict) and obj.get("prompt"):
            logger.info(
                f"Agent: Generating image for object {i+1}: {obj['prompt']}"
            )
            obj_name = obj.get("name", f"object_{i+1}").replace(" ", "_").lower()
            filename = obj_name + ".png"
            generate_image(obj["prompt"], filename)
        else:
            logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
            logger.info(f"\n[Skipping object {i+1} - missing prompt]")

    logger.info("\nAgent: Image generation process complete.")
    return f"Image generation process complete."
