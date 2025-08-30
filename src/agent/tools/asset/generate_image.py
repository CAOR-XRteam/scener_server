from colorama import Fore
from langchain_core.tools import tool, BaseTool
from lib import logger
from model import black_forest
from sdk.scene import ImprovedDecompositionOutput
import os
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from typing import Type


class ImageMetaData(BaseModel):
    id: str
    prompt: str
    filename: str
    path: str
    error: str | None


class GenerateImageOutput(BaseModel):
    action: Literal["image_generation"]
    message: str
    generated_images_data: list[ImageMetaData]


class GenerateImageOutputWrapper(BaseModel):
    general_image_output: GenerateImageOutput


class GenerateImageToolInput(BaseModel):
    improved_decomposition: dict = Field(
        description="The JSON representing the decomposed scene."
    )


@tool(args_schema=GenerateImageToolInput)
def generate_image(
    improved_decomposition: dict,
) -> GenerateImageOutput:
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""
    try:
        validated_data = ImprovedDecompositionOutput(**improved_decomposition)
    except ValidationError as e:
        logger.error(f"Pydantic validation failed for improver payload: {e}")
        raise ValueError(f"Invalid payload structure for improver tool. Details: {e}")

    logger.info(f"\nDecomposed scene received: {validated_data}. Generating image...")

    # Retrieve list of to-generated objects
    objects_to_generate = validated_data.scene_data.scene.objects
    logger.info(f"Decomposed objects to generate: {objects_to_generate}")

    if not objects_to_generate:
        logger.info(
            "The decomposition resulted in no specific objects to generate images for."
        )
        return "[No objects to generate images for.]"

    generated_images_data = []
    successful_images = 0

    # Objects generation
    for i, obj in enumerate(objects_to_generate):
        obj_prompt = obj.prompt
        if obj_prompt:
            logger.info(f"Generating image for object {i+1}: {obj_prompt}")
            obj_id = obj.id
            dir_path = str(Path(__file__).resolve().parents[3] / "media" / "temp")
            os.makedirs(dir_path, exist_ok=True)
            filename = dir_path + "/" + obj_id + ".png"
            try:
                black_forest.generate(obj_prompt, filename)

                generated_images_data.append(
                    ImageMetaData(
                        id=obj_id,
                        prompt=obj_prompt,
                        filename=f"{obj_id}.png",
                        path=filename,
                        error=None,
                    )
                )
                successful_images += 1
            except Exception as e:
                logger.error(f"Failed to generate image:{e}")
                generated_images_data.append(
                    ImageMetaData(
                        id=obj_id,
                        prompt=obj_prompt,
                        filename=f"{obj_id}.png",
                        path=filename,
                        error=f"Failed to generate image: {e}",
                    )
                )
                pass
        else:
            logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
            logger.info(f"\n[Skipping object {i+1} - missing prompt]")

    logger.info("\nImage generation process complete.")

    return GenerateImageOutputWrapper(
        general_image_output=GenerateImageOutput(
            action="image_generation",
            message=f"Image generation process complete. Generated {successful_images} from {len(objects_to_generate)} images.",
            generated_images_data=generated_images_data,
        )
    )


# TODO: modify
if __name__ == "__main__":
    scene_dict = {
        "scene": {
            "objects": [
                {
                    "id": "cream_couch",
                    "type": "furniture",
                    "material": "plush_fabric",
                    "prompt": "A plush, cream-colored couch with a low back and rolled arms, front camera view, placed on a white and empty background, completely detached from its surroundings.",
                },
                {
                    "id": "gray_cat",
                    "type": "prop",
                    "material": "glossy_fur",
                    "prompt": "A sleek, gray cat with bright green eyes, front camera view, placed on a white and empty background, completely detached from its surroundings.",
                },
                {
                    "id": "living_room",
                    "type": "room",
                    "material": "polished_wood",
                    "prompt": "A squared room, room view from the outside with a distant 3/4 top-down perspective, placed on a white and empty background, completely detached from its surroundings.",
                },
            ]
        }
    }

    res = generate_image.invoke(
        {"improved_decomposition": scene_dict}
    )  # ✅ Pass a dict
    print(res)
