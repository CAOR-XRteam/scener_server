from beartype import beartype
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from agent.tools.pipeline.basic import decompose_and_improve
from agent.tools.pipeline.td_object_generation import generate_3d_object
from agent.tools.scene.decomposer import FinalDecomposer
from lib import logger, load_config
from typing import Generator
from sdk.scene import Scene, SceneObject, SceneComponent


def iter_scene_objects_from_scene(scene: Scene) -> Generator[SceneObject, None, None]:
    for root_obj in scene.graph:
        yield from iter_scene_objects(root_obj)


def iter_scene_objects(obj: SceneObject) -> Generator[SceneObject, None, None]:
    yield obj
    for child in obj.children:
        yield from iter_scene_objects(child)


class Generate3DScenetoolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's description prompt to generate images from."
    )


@beartype
@tool(args_schema=Generate3DScenetoolInput)
def generate_3d_scene(user_input: str):
    try:
        decomposed_input = decompose_and_improve(user_input)
    except Exception as e:
        logger.error(f"Failed to decompose and improve input: {e}")
        raise ValueError(f"Failed to decompose and improve input: {e}")

    try:
        config = load_config()
        final_decomposer_model_name = config.get("final_decomposer_model")

        final_decomposer = FinalDecomposer(final_decomposer_model_name)
        final_decomposition = final_decomposer.decompose(user_input, decomposed_input)
    except Exception as e:
        logger.error(f"Failed to do the final decomposition: {e}")
        raise ValueError(f"Failed to do the final decomposition: {e}")

    for obj in iter_scene_objects_from_scene(final_decomposition):
        print(obj.id, obj.position, obj.components)
