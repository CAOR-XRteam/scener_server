from beartype import beartype
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from agent.tools.pipeline.td_object_generation import (
    TDObjectMetaData,
    generate_3d_object_from_prompt,
)
from agent.tools.scene.decomposer import (
    final_decomposition,
    initial_decomposition,
    FinalDecompositionOutput,
)
from lib import logger, load_config
from typing import Generator
from sdk.scene import Scene, SceneObject, SceneComponent


class Generate3DSceneToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


class Generate3DSceneOutput(BaseModel):
    text: str
    final_decomposition: Scene
    objects_to_send: list[TDObjectMetaData]


class Generate3DSceneOutputWrapper(BaseModel):
    generate_3d_scene_output: Generate3DSceneOutput


@tool(args_schema=Generate3DSceneToolInput)
@beartype
def generate_3d_scene(user_input: str) -> dict:
    """
    Use this to create a complete 3D environment or scene with multiple objects or a background.
    Examples: 'a cat and a dog in a room', 'a car on a road'.
    This is the correct choice for any 3D request that is NOT a single, isolated object.
    """
    try:
        initial_decomposition_output = initial_decomposition(user_input)
    except Exception as e:
        logger.error(f"Failed to decompose input: {e}")
        raise ValueError(f"Failed to decompose input: {e}")

    objects_to_send = []

    try:
        for object in initial_decomposition_output.scene.objects:
            if object.type == "dynamic":
                objects_to_send.append(
                    generate_3d_object_from_prompt(object.prompt, object.id)
                )
    except:
        logger.error(f"Failed to generate 3D object: {e}")
        raise ValueError(f"Failed to generate 3D object: {e}")

    try:
        final_decomposition_output = final_decomposition(
            user_input, initial_decomposition_output
        )

        return Generate3DSceneOutputWrapper(
            generate_3d_scene_output=Generate3DSceneOutput(
                text=f"Generated 3D scene for {user_input}",
                final_decomposition=final_decomposition_output.scene,
                objects_to_send=objects_to_send,
            )
        ).model_dump()
    except Exception as e:
        logger.error(f"Failed to do the final decomposition: {e}")
        raise ValueError(f"Failed to do the final decomposition: {e}")
