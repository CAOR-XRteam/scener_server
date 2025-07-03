from beartype import beartype
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from agent.tools.pipeline.td_object_generation import (
    TDObjectMetaData,
    generate_3d_object_from_prompt,
)
from agent.tools.scene.decomposer import (
    FinalDecomposer,
    InitialDecomposer,
    FinalDecompositionOutput,
)
from lib import logger, load_config
from typing import Generator
from sdk.scene import ComponentType, Scene, SceneObject, SceneComponent


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
    # config = load_config()
    # try:
    #     initial_decomposer_model_name = config.get("initial_decomposer_model")
    #     initial_decomposition = InitialDecomposer(
    #         initial_decomposer_model_name
    #     ).decompose(user_input)

    # except Exception as e:
    #     logger.error(f"Failed to decompose input: {e}")
    #     raise ValueError(f"Failed to decompose input: {e}")

    # objects_to_send = []

    # try:
    #     for object in initial_decomposition.scene.objects:
    #         if object.type == "dynamic":
    #             objects_to_send.append(generate_3d_object_from_prompt(object.prompt))
    # except:
    #     logger.error(f"Failed to generate 3D object: {e}")
    #     raise ValueError(f"Failed to generate 3D object: {e}")

    try:
        # config = load_config()
        # final_decomposer_model_name = config.get("final_decomposer_model")

        # final_decomposition = FinalDecomposer(final_decomposer_model_name).decompose(
        #     user_input, initial_decomposition
        # )

        data = {
            "name": "A Room with a Dog and Cat",
            "skybox": {
                "type": "gradient",
                "color1": {"r": 0.8, "g": 0.8, "b": 0.8, "a": 1.0},
                "color2": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
                "up_vector": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0},
                "intensity": 1.0,
                "exponent": 2.0,
            },
            "graph": [
                {
                    "id": "room_770704",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 10.0, "y": 5.0, "z": 10.0},
                    "components": [
                        {
                            "component_type": "primitive",
                            "shape": "cube",
                            "color": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},
                        }
                    ],
                    "children": [
                        {
                            "id": "dog_container",
                            "position": {"x": -5.0, "y": 0.0, "z": 0.0},
                            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                            "components": [
                                {"component_type": "dynamic", "id": "dog_df0d92"}
                            ],
                            "children": [],
                        },
                        {
                            "id": "cat_container",
                            "position": {"x": 5.0, "y": 0.0, "z": 0.0},
                            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                            "components": [
                                {"component_type": "dynamic", "id": "cat_947a70"}
                            ],
                            "children": [],
                        },
                    ],
                }
            ],
        }

        import json

        json_str = json.dumps(data)

        return Generate3DSceneOutputWrapper(
            generate_3d_scene_output=Generate3DSceneOutput(
                text=f"Generated 3D scene for {user_input}",
                final_decomposition=Scene.model_validate_json(json_str),
                objects_to_send=[
                    TDObjectMetaData(
                        id="a9253727-cf24-4236-a8e1-d16b1927f8f1",
                        filename="a9253727-cf24-4236-a8e1-d16b1927f8f1.glb",
                        path="/home/xrteam/Desktop/Dev/Scener/src/media/temp/a9253727-cf24-4236-a8e1-d16b1927f8f1.glb",
                        error=None,
                    ),
                    TDObjectMetaData(
                        id="6d80a040-c300-4bf1-b7de-0c9a59be9ee3",
                        filename="6d80a040-c300-4bf1-b7de-0c9a59be9ee3.glb",
                        path="/home/xrteam/Desktop/Dev/Scener/src/media/temp/6d80a040-c300-4bf1-b7de-0c9a59be9ee3.glb",
                        error=None,
                    ),
                ],
            )
        ).model_dump()
    except Exception as e:
        logger.error(f"Failed to do the final decomposition: {e}")
        raise ValueError(f"Failed to do the final decomposition: {e}")
