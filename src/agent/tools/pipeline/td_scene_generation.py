from beartype import beartype
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from agent.tools.pipeline.basic import decompose_and_improve
from agent.tools.pipeline.td_object_generation import generate_3d_object_from_prompt
from agent.tools.scene.decomposer import FinalDecomposer, InitialDecomposer
from lib import logger, load_config
from typing import Generator
from sdk.scene import Scene, SceneObject, SceneComponent


class Generate3DSceneToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


@beartype
@tool(args_schema=Generate3DSceneToolInput)
def generate_3d_scene(user_input: str):
    config = load_config()
    try:
        initial_decomposer_model_name = config.get("initial_decomposer_model")
        initial_decomposer = InitialDecomposer(initial_decomposer_model_name)
        initial_decomposition = initial_decomposer.decompose(user_input)
    except Exception as e:
        logger.error(f"Failed to decompose input: {e}")
        raise ValueError(f"Failed to decompose input: {e}")

    try:
        for object in initial_decomposition.scene.objects:
            if object.type == "dynamic":
                generate_3d_object_from_prompt(object.prompt)
    except:
        logger.error(f"Failed to generate 3D object: {e}")
        raise ValueError(f"Failed to generate 3D object: {e}")

    try:
        config = load_config()
        final_decomposer_model_name = config.get("final_decomposer_model")

        final_decomposer = FinalDecomposer(final_decomposer_model_name)
        return final_decomposer.decompose(user_input, initial_decomposition)
    except Exception as e:
        logger.error(f"Failed to do the final decomposition: {e}")
        raise ValueError(f"Failed to do the final decomposition: {e}")
