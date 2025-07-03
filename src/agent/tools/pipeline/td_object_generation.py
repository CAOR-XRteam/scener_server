import os

from beartype import beartype
from colorama import Fore
from langchain_core.tools import tool
from pathlib import Path
from pydantic import BaseModel, Field

from agent.tools.pipeline.image_generation import generate_image_from_prompt
from lib import logger
from model import trellis

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

        trellis.generate(image_meta_data.path, image_meta_data.id)

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
def generate_3d_object(user_input: str) -> dict:
    """Generate 3D object from user's prompt"""
    data = generate_3d_object_from_prompt(user_input)
    return Generate3DObjectOutputWrapper(
        generate_3d_object_output=Generate3DObjectOutput(
            text=f"Generated 3D object for '{user_input}'", data=data
        )
    ).model_dump()


if __name__ == "__main__":
    user_input = "big black cat on a table"
    res = generate_3d_object(user_input)
    print(res)
