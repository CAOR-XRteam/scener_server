from colorama import Fore
from langchain_core.tools import tool
from lib import logger
from model import black_forest
import os
from pathlib import Path
from pydantic import BaseModel, Field
from agent.tools.pipeline.basic import decompose_and_improve
from beartype import beartype


class ImageMetaData(BaseModel):
    id: str
    prompt: str
    filename: str
    path: str
    error: str | None


class GenerateImageOutput(BaseModel):
    text: str
    data: list[ImageMetaData]


class GenerateImageOutputWrapper(BaseModel):
    generate_image_output: GenerateImageOutput


class GenerateImageToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


@tool(args_schema=GenerateImageToolInput)
@beartype
def generate_image(
    user_input: str,
) -> GenerateImageOutputWrapper:
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""
    logger.info(
        f"\nReceived user input for image generation: {Fore.GREEN}{user_input}{Fore.RESET}"
    )

    try:
        improved_and_decomposed_input = decompose_and_improve(user_input)
    except Exception as e:
        logger.error(f"Failed to decompose and improve input: {e}")
        raise ValueError(f"Failed to decompose and improve input: {e}")

    # Retrieve list of to-generate objects
    objects_to_generate = improved_and_decomposed_input.scene.objects
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
        generate_image_output=GenerateImageOutput(
            text=f"Generated {len(generated_images_data)} of {len(objects_to_generate)} images.",
            data=generated_images_data,
        )
    )


if __name__ == "__main__":
    user_input = "Big black cat on a table"
    res = generate_image(user_input)
    print(res)
