from beartype import beartype
from colorama import Fore
from langchain_core.tools import tool
from pathlib import Path
from pydantic import BaseModel, Field
from uuid import uuid4

from agent.tools.pipeline.basic import decompose_and_improve
from agent.tools.scene.improver import Improver
from lib import logger, load_config
from model import stable_diffusers


class ImageMetaData(BaseModel):
    id: str
    prompt: str
    filename: str
    path: Path
    error: str | None


class GenerateImageOutput(BaseModel):
    text: str
    data: ImageMetaData


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
def generate_image_from_prompt(prompt: str, id: str | None = None) -> ImageMetaData:
    if not id:
        id = uuid4()

    logger.info(f"Generating image for '{id}': {prompt[:10]}...")

    output_dir = Path(__file__).resolve().parents[3] / "media" / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{id}.png"

    try:
        stable_diffusers.generate(prompt, str(output_path))

        return ImageMetaData(
            id=str(id),
            prompt=prompt,
            filename=output_path.name,
            path=output_path,
            error=None,
        )
    except Exception as e:
        logger.error(f"Failed to generate image for '{id}': {e}", exc_info=True)
        return ImageMetaData(
            id=id,
            prompt=prompt,
            filename=output_path.name,
            path=output_path,
            error=str(e),
        )


@tool(args_schema=GenerateImageToolInput)
@beartype
def generate_image(user_input: str):
    """Generates an image from user's prompt"""
    data = generate_image_from_prompt(user_input)

    return GenerateImageOutputWrapper(
        GenerateImageOutput(text=f"Generated image for {user_input}", data=data)
    )


# @tool(args_schema=GenerateImageToolInput)
# @beartype
# def generate_image(
#     user_input: str,
# ) -> GenerateImageOutputWrapper:
#     logger.info(
#         f"\nReceived user input for image generation: {Fore.GREEN}{user_input}{Fore.RESET}"
#     )

#     try:
#         decomposed_input = decompose_and_improve(user_input)
#     except Exception as e:
#         logger.error(f"Failed to decompose and improve input: {e}")
#         raise ValueError(f"Failed to decompose and improve input: {e}")

#     objects_to_generate = decomposed_input.scene.objects

#     if not objects_to_generate:
#         logger.warning("Decomposition resulted in no objects to generate.")
#         return GenerateImageOutput(
#             text="No images generated from this request,", data=[]
#         )

#     data = []
#     successful_images = 0
#     output_dir = Path(__file__).resolve().parents[3] / "media" / "temp"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for obj in objects_to_generate:
#         if not obj.prompt:
#             logger.warning(f"Skipping object '{obj.id}' due to missing prompt.")
#             continue

#         logger.info(f"Generating image for '{obj.id}': {obj.prompt[:80]}...")
#         output_path = output_dir / f"{obj.id}.png"

#         try:
#             black_forest.generate(obj.prompt, str(output_path))

#             data.append(
#                 ImageMetaData(
#                     id=obj.id,
#                     prompt=obj.prompt,
#                     filename=output_path.name,
#                     path=str(output_path),
#                     error=None,
#                 )
#             )
#             successful_images += 1
#         except Exception as e:
#             logger.error(f"Failed to generate image for '{obj.id}': {e}", exc_info=True)
#             data.append(
#                 ImageMetaData(
#                     id=obj.id,
#                     prompt=obj.prompt,
#                     filename=output_path.name,
#                     path=str(output_path),
#                     error=str(e),
#                 )
#             )

#     logger.info("Image generation process complete.")

#     return GenerateImageOutputWrapper(
#         generate_image_output=GenerateImageOutput(
#             text=f"Generated {successful_images} of {len(objects_to_generate)} images.",
#             data=data,
#         )
#     )


# @tool(args_schema=GenerateImageToolInput)
# @beartype
# def generate_image(user_input: str) -> GenerateImageOutputWrapper:
#     """Generates an image based on the decomposed user's prompt using the Black Forest model."""
#     try:
#         return _generate_image(user_input)
#     except Exception as e:
#         logger.error(f"Error in generate_image tool wrapper: {e}")
#         raise


if __name__ == "__main__":
    user_input = "Big black cat on a table"
    res = generate_image(user_input)
    print(res)
