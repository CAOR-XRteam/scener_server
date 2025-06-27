from beartype import beartype
from colorama import Fore
from langchain_core.tools import tool
from lib import logger
from model import trellis
from pathlib import Path
from pydantic import BaseModel, Field

from agent.tools.pipeline.image_generation import _generate_image


class TDObjectMetaData(BaseModel):
    id: str
    filename: str
    path: str
    error: str


class Generate3DObjectOutput(BaseModel):
    text: str
    data: list[TDObjectMetaData]


class Generate3DObjectOutputWrapper(BaseModel):
    generate_3d_object_output: Generate3DObjectOutput


class Generate3dObjectToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


@beartype
def _generate_3d_object(
    user_input: str,
) -> Generate3DObjectOutputWrapper:
    logger.info(
        f"\nReceived user input for image generation: {Fore.GREEN}{user_input}{Fore.RESET}"
    )

    try:
        generate_image_output = _generate_image(user_input)
    except Exception as e:
        logger.error(f"Failed to generate images from user's input: {e}")
        raise ValueError(f"Failed to generate images from user's input: {e}")

    image_meta_data = generate_image_output.generate_image_output.data
    data = []
    successful_objects = 0

    media_temp_dir = Path(__file__).resolve().parents[3] / "media" / "temp"
    media_temp_dir.mkdir(parents=True, exist_ok=True)

    for image_meta in image_meta_data:
        if image_meta.error:
            logger.warning(f"Skipping 3D generation for '{image_meta.id}': {e}")
            continue

        logger.info(f"Generating 3D object from image: {image_meta.path}")

        output_path = media_temp_dir / f"{image_meta.id}.glb"

        try:
            trellis.generate(image_meta.path, image_meta.id)

            data.append(
                TDObjectMetaData(
                    id=image_meta.id,
                    filename=f"{image_meta.id}.glb",
                    path=str(output_path),
                    error=None,
                )
            )
            successful_objects += 1
        except Exception as e:
            logger.error(
                f"Failed to generate 3D object for '{image_meta.id}': {e}",
                exc_info=True,
            )
            data.append(
                TDObjectMetaData(
                    id=image_meta.id,
                    filename=f"{image_meta.id}.glb",
                    path=str(output_path),
                    error=str(e),
                )
            )

    logger.info("3D object generation process complete.")

    return Generate3DObjectOutputWrapper(
        generate_3d_object_output=Generate3DObjectOutput(
            text=f"Generated {successful_objects} of {len(image_meta_data)} 3D objects.",
            data=data,
        )
    )


@tool(args_schema=Generate3dObjectToolInput)
@beartype
def generate_3d_object(user_input: str) -> Generate3DObjectOutputWrapper:
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""
    logger.info(
        f"\nTool 'generate_3d_object' triggered with input: {Fore.GREEN}{user_input}{Fore.RESET}"
    )
    try:
        return _generate_3d_object(user_input)
    except Exception as e:
        logger.error(f"Failed to generate 3d objects: {e}")
        raise


if __name__ == "__main__":
    user_input = "big black cat on a table"
    res = generate_3d_object(user_input)
    print(res)
