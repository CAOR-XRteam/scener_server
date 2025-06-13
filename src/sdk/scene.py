from pydantic import BaseModel, model_validator
from typing import Literal, Union

# TODO: change literals to enums, more constraints on fields with Field


class DecomposedObject(BaseModel):
    id: str
    name: str
    type: str
    material: str
    prompt: str


class InitialDecompositionData(BaseModel):
    objects: list[DecomposedObject]


class InitialDecomposition(BaseModel):
    scene: InitialDecompositionData


class InitialDecompositionOutput(BaseModel):
    decomposition: InitialDecomposition
    original_user_prompt: str


class ImprovedDecompositionOutput(BaseModel):
    decomposition: InitialDecomposition
    original_user_prompt: str


class ToolOutputWrapper(BaseModel):
    output: InitialDecompositionOutput | ImprovedDecompositionOutput


class ColorRGBA(BaseModel):
    r: float
    g: float
    b: float
    a: float | None


class Vector3(BaseModel):
    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class Vector4(BaseModel):
    x: float
    y: float
    z: float
    w: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.w]


class SceneObject(BaseModel):
    id: str
    name: str
    # TODO: do we need predefined objects (prefabs) or treillis-generated objects are enough? colliders? materials?
    type: Literal["dynamic", "primitive"]
    position: Vector3
    rotation: Vector3
    scale: Vector3
    path: str | None
    shape: Literal["cube", "sphere", "capsule", "cylinder", "plane", "quad"] | None

    @model_validator(mode="after")
    def check_conditional_fields(self):
        if self.type == "primitive" and self.shape is None:
            raise ValueError("shape must be set for primitive objects")
        if self.type == "dynamic" and self.id is None:
            raise ValueError("id must be set for dynamic objects")
        return self


class GradientSkybox(BaseModel):
    type: Literal["gradient"]
    color1: ColorRGBA
    color2: ColorRGBA
    up_vector: Vector4
    intensity: float
    exponent: float


class SunSkybox(BaseModel):
    type: Literal["sun"]
    top_color: ColorRGBA
    top_exponent: float
    horizon_color: ColorRGBA
    bottom_color: ColorRGBA
    bottom_exponent: float
    sky_intensity: float
    sun_color: ColorRGBA
    sun_intensity: float
    sun_alpha: float
    sun_beta: float
    sun_vector: Vector4


class CubedSkybox(BaseModel):
    type: Literal["cubed"]
    tint_color: ColorRGBA
    exposure: float
    rotation: float
    cube_map: str


Skybox = Union[GradientSkybox, SunSkybox, CubedSkybox]


class BaseLight(BaseModel):
    id: str
    position: Vector3
    rotation: Vector3
    scale: Vector3
    color: ColorRGBA
    intensity: float
    indirect_multiplier: float


class SpotLight(BaseLight):
    type: Literal["spot"]
    range: float
    spot_angle: float
    mode: Literal["baked", "mixed", "realtime"]
    shadow_type: Literal["no_shaows", "hard_shadows", "soft_shadows"]


class DirectionalLight(BaseLight):
    type: Literal["directional"]
    mode: Literal["baked", "mixed", "realtime"]
    shadow_type: Literal["no_shaows", "hard_shadows", "soft_shadows"]


class PointLight(BaseLight):
    type: Literal["point"]
    range: float
    mode: Literal["baked", "mixed", "realtime"]
    shadow_type: Literal["no_shaows", "hard_shadows", "soft_shadows"]


class AreaLight(BaseLight):
    type: Literal["area"]
    shape: Literal["rectangle", "disk"]
    range: float
    width: float | None
    height: float | None
    radius: float | None

    @model_validator(mode="after")
    def check_conditional_fields(self):
        if self.shape == "rectangle" and self.width is None or self.height is None:
            raise ValueError("width and height must be set for rectangle area light")
        if self.shape == "disk" and self.radius is None:
            raise ValueError("radius must be set for disk area light")
        return self


Light = Union[SpotLight, DirectionalLight, PointLight, AreaLight]


class Scene(BaseModel):
    skybox: Skybox
    lights: list[Light]
    objects: list[SceneObject]


class FinalDecompositionOutput(BaseModel):
    action: Literal["scene_generation"]
    message: Literal["Scene description has been successfully generated."]
    final_scene_json: Scene
