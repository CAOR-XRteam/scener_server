from colorama import Fore
from langchain_core.tools import tool
from lib import logger
from model import trellis
from sdk.scene import InitialDecompositionOutput
from agent.tools.asset.generate_image import ImageMetaData, GenerateImageOutput
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal


class TDObjectMetaData(BaseModel):
    id: str
    filename: str
    path: str
    error: str


class Generate3DObjectOutput(BaseModel):
    action: Literal["3d_object_generation"]
    message: str
    generated_images_data: list[TDObjectMetaData]


class Generate3dObjectToolInput(BaseModel):
    image_generation_result: dict = Field(
        description="The JSON containing the result of 'generate_image' tool call"
    )


@tool(args_schema=Generate3dObjectToolInput)
def generate_3d_object(
    image_generation_result: GenerateImageOutput,
) -> Generate3DObjectOutput:
    """Generates an image based on the decomposed user's prompt using the Black Forest model."""
    image_data = image_generation_result.generated_images_data
    generated_objects_data = []
    successful_objects = 0

    # Objects generation
    for i, obj in enumerate(image_data):
        logger.info(f"Generating 3D object from image {obj.id}")
        obj_id = obj.id
        dir_path = str(Path(__file__).resolve().parents[3] / "media" / "temp")
        os.makedirs(dir_path, exist_ok=True)
        filename = dir_path + "/" + obj_id + ".glb"
        try:
            trellis.generate(obj.path, obj.id)

            generated_objects_data.append(
                TDObjectMetaData(
                    id=obj_id,
                    filename=f"{obj_id}.glb",
                    path=filename,
                    error=None,
                )
            )
            successful_objects += 1
        except Exception as e:
            logger.error(f"Failed to generate image:{e}")
            generated_objects_data.append(
                TDObjectMetaData(
                    id=obj_id,
                    filename=f"{obj_id}.glb",
                    path=filename,
                    error=None,
                )
            )
            pass

    logger.info("3D object generation process complete.")

    return Generate3DObjectOutput(
        action="3d_object_generation",
        message=f"3D object generation process complete. Generated {successful_objects} 3d objects from {len(image_data)} images.",
        generated_images_data=generated_objects_data,
    )


# TODO: modify
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

    res = generate_3d_object.invoke(
        {"improved_decomposed_input": scene_dict}
    )  # ✅ Pass a dict
    print(res)
