import os

from beartype import beartype
from colorama import Fore
from langchain_core.tools import tool
from pathlib import Path
from pydantic import BaseModel, Field

from agent.tools.pipeline.image_generation import generate_image_from_prompt
from lib import logger
from model import hunyuan

# TODO: modify the tool so that it doesn't reload model on every new request


class TDObjectMetaData(BaseModel):
    id: str
    filename: str
    path: str
    error: str | None


class Generate3DObjectOutput(BaseModel):
    text: str
    data: TDObjectMetaData


# Langchain tool implementation doesn't work well with pydantic models when they have several fields
# (it converts output to string with invalid format that isn't convertible to pydantic model or even json)
# so we creating a wrapper around output structure to workaround


class Generate3DObjectOutputWrapper(BaseModel):
    generate_3d_object_output: Generate3DObjectOutput


class Generate3DObjectToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


@beartype
def generate_3d_object_from_prompt(prompt: str) -> TDObjectMetaData:
    try:
        image_meta_data = generate_image_from_prompt(prompt)

        logger.info(f"Generating 3D object from image: {image_meta_data.path}")

        hunyuan.generate(image_meta_data.path, image_meta_data.id)

        return TDObjectMetaData(
            id=image_meta_data.id,
            filename=f"{image_meta_data.id}.glb",
            path=str(image_meta_data.path.parent / f"{image_meta_data.id}.glb"),
            error=None,
        )

    except Exception as e:
        logger.error(f"Failed to generate 3D object for '{image_meta_data.id}': {e}")

        return TDObjectMetaData(
            id=image_meta_data.id,
            filename=f"{image_meta_data.id}.glb",
            path=str(image_meta_data.path.parent / f"{image_meta_data.id}.glb"),
            error=str(e),
        )


@tool(args_schema=Generate3DObjectToolInput)
@beartype
def generate_3d_object(user_input: str) -> Generate3DObjectOutputWrapper:
    """Generate 3D object from user's prompt"""
    data = generate_3d_object_from_prompt(user_input)
    return Generate3DObjectOutputWrapper(
        generate_3d_object_output=Generate3DObjectOutput(
            text=f"Generated 3D object for '{user_input}'", data=data
        )
    )


# @tool(args_schema=Generate3DObjectToolInput)
# @beartype
# def generate_3d_object(
#     user_input: str,
# ) -> Generate3DObjectOutputWrapper:
#     """Generates an image based on the decomposed user's prompt using the Black Forest model."""
#     logger.info(
#         f"\nReceived user input for image generation: {Fore.GREEN}{user_input}{Fore.RESET}"
#     )

#     try:
#         generate_image_output = _generate_image(user_input)
#     except Exception as e:
#         logger.error(f"Failed to generate images from user's input: {e}")
#         raise ValueError(f"Failed to generate images from user's input: {e}")

#     image_meta_data = generate_image_output.generate_image_output.data
#     data = []
#     successful_objects = 0

#     media_temp_dir = Path(__file__).resolve().parents[3] / "media" / "temp"
#     media_temp_dir.mkdir(parents=True, exist_ok=True)

#     for image_meta in image_meta_data:
#         if image_meta.error:
#             logger.warning(f"Skipping 3D generation for '{image_meta.id}': {e}")
#             continue

#         data.append(
#             generate_3d_object_from_image(
#                 image_meta_data.path,
#                 image_meta_data.id,
#                 media_temp_dir / f"{image_meta.id}.glb",
#             )
#         )

#     logger.info("3D object generation process complete.")

#     return Generate3DObjectOutputWrapper(
#         generate_3d_object_output=Generate3DObjectOutput(
#             text=f"Generated {successful_objects} of {len(image_meta_data)} 3D objects.",
#             data=data,
#         )
#     )


if __name__ == "__main__":
    user_input = "big black cat on a table"
    res = generate_3d_object(user_input)
    print(res)
