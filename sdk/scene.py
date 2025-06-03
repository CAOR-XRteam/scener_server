from pydantic import BaseModel
from typing import Literal

# TODO: change literals to enums


class Vector3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class SceneObject(BaseModel):
    id: str
    name: str
    type: Literal["predefined", "dynamic"]
    position: Vector3
    rotation: Vector3
    scale: Vector3
    path: str | None


class Skybox(BaseModel):
    type: Literal["gradient", "horizontal_with_sun", "cubed"]


class Scene(BaseModel):
    skybox: Skybox | None
    objects: list[SceneObject]
