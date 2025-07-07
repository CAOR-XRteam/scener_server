from beartype import beartype
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tools.scene.analyzer import analyze
from agent.tools.pipeline.td_object_generation import (
    TDObjectMetaData,
    generate_3d_object_from_prompt,
)
from lib import logger
from sdk.scene import Scene


class Modify3DSceneToolInput(BaseModel):
    user_input: str = (Field(description="The raw user's modification request."),)
    json_scene: Scene = Field(Field(description="The JSON representing current scene."))


class Modify3DSceneOutput(BaseModel):
    text: str
    modified_scene: Scene
    objects_to_send: list[TDObjectMetaData]


@tool(args_schema=Modify3DSceneToolInput)
@beartype
def modify_3d_scene(user_input: str, json_scene: Scene) -> dict:
    """Creates a complete 3D environment or scene with multiple objects or a background."""
    logger.log(f"Modifying 3D scene from prompt: {user_input[:10]}...")

    try:
        analysis_output = analyze(user_input, json_scene)
    except Exception:
        raise

    objects_to_send = []

    try:
        for object in analysis_output.regenerations:
            generated_object_meta_data = generate_3d_object_from_prompt(
                object.prompt, object.id
            )
            generated_object_meta_data.id = object.id
            objects_to_send.append(generated_object_meta_data)
    except Exception:
        raise

    return Modify3DSceneOutput(
        text=f"Scene modification for {user_input}",
        modified_scene=analysis_output.final_graph,
        objects_to_send=objects_to_send,
    ).model_dump()
