from beartype import beartype
from colorama import Fore
from langchain_core.tools import tool
from lib import logger
from model import black_forest
from pathlib import Path
from pydantic import BaseModel, Field

from agent.tools.pipeline.basic import decompose_and_improve


class ImageMetaData(BaseModel):
    id: str
    prompt: str
    filename: str
    path: str
    error: str | None


class GenerateImageOutput(BaseModel):
    text: str
    data: list[ImageMetaData]


# Langchain tool implementation doesn't work well with pydantic models when they have several fields
# (it converts output to string with invalid format that isn't convertible to pydantic model or even json)
# so we creating a wrapper around output structure to workaround


class GenerateImageOutputWrapper(BaseModel):
    generate_image_output: GenerateImageOutput


class GenerateImageToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


@beartype
def _generate_image(
    user_input: str,
) -> GenerateImageOutputWrapper:
    logger.info(
        f"\nReceived user input for image generation: {Fore.GREEN}{user_input}{Fore.RESET}"
    )

    try:
        decomposed_input = decompose_and_improve(user_input)
    except Exception as e:
        logger.error(f"Failed to decompose and improve input: {e}")
        raise ValueError(f"Failed to decompose and improve input: {e}")

    objects_to_generate = decomposed_input.scene.objects

    if not objects_to_generate:
        logger.warning("Decomposition resulted in no objects to generate.")
        return GenerateImageOutput(
            text="No images generated from this request,", data=[]
        )

    data = []
    successful_images = 0
    output_dir = Path(__file__).resolve().parents[3] / "media" / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    for obj in objects_to_generate:
        if not obj.prompt:
            logger.warning(f"Skipping object '{obj.id}' due to missing prompt.")
            continue

        logger.info(f"Generating image for '{obj.id}': {obj.prompt[:80]}...")
        output_path = output_dir / f"{obj.id}.png"

        try:
            black_forest.generate(obj.prompt, str(output_path))

            data.append(
                ImageMetaData(
                    id=obj.id,
                    prompt=obj.prompt,
                    filename=output_path.name,
                    path=str(output_path),
                    error=None,
                )
            )
            successful_images += 1
        except Exception as e:
            logger.error(f"Failed to generate image for '{obj.id}': {e}", exc_info=True)
            data.append(
                ImageMetaData(
                    id=obj.id,
                    prompt=obj.prompt,
                    filename=output_path.name,
                    path=str(output_path),
                    error=str(e),
                )
            )

    logger.info("Image generation process complete.")

    return GenerateImageOutputWrapper(
        generate_image_output=GenerateImageOutput(
            text=f"Generated {successful_images} of {len(objects_to_generate)} images.",
            data=data,
        )
    )


@tool(args_schema=GenerateImageToolInput)
@beartype
def generate_image(user_input: str) -> GenerateImageOutputWrapper:
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""
    try:
        return _generate_image(user_input)
    except Exception as e:
        logger.error(f"Error in generate_image tool wrapper: {e}")
        raise


if __name__ == "__main__":
    user_input = "Big black cat on a table"
    res = generate_image(user_input)
    print(res)
