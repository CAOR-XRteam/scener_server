import json
import uuid

from agent.llm.creation import initialize_model
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from lib import logger
from pydantic import BaseModel, Field, ValidationError
from sdk.scene import *


class InitialDecomposerToolInput(BaseModel):
    user_input: str = Field(
        description="The raw user's scene description prompt to be decomposed."
    )


@beartype
class InitialDecomposer:
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.system_prompt = """
You are a highly specialized and precise Scene Decomposer for a 3D rendering workflow. Your sole task is to accurately convert a scene description string into structured JSON, adhering to strict rules. The output must always extract **verbatim zero-shot prompts** for each object in the scene, following the format provided below.

YOUR CRITICAL TASK:
- Decompose the scene description into several distinct elements, ensuring at least one 'room' object.
- Convert the scene into a valid JSON object.
- Focus ONLY on **key physical elements**. **Do NOT extract minor details** or interpret the scene beyond identifying these key elements.
- **The most critical part of the output is the 'prompt' field for each object.** This field must contain the **exact, verbatim phrase** describing that specific object as it appeared in the user's input scene description. **Do NOT modify, enhance, or add ANY details (like camera angles, background information, or context) to this user-provided object description.**

OUTPUT FORMAT:
- Output **ONLY** the JSON. **NO explanations**, **NO markdown**, **NO extra formatting**.

JSON STRUCTURE (STRICT AND REQUIRED):
{{
  "scene": {{
    "objects": [
      {{
        "id": "unique_object_id" // must be str
        "name": "concise_descriptive_object_name",  // short, clear identifier (e.g., 'black_cat', 'wooden_table')
        "type": "object_category",  // one of: 'mesh', 'furniture', 'prop', 'room'
        "material": "primary_material_name",  // e.g., 'polished_wood', 'glossy_fur', 'ceramic'
        "prompt": "The exact, verbatim phrase describing this specific object, extracted directly from the user's input scene description. This must be the exact noun phrase or descriptive phrase from the input that **identifies or describes the object itself**. **Do NOT include prepositions (like 'on', 'in', 'under'), verbs describing actions, or phrases indicating relationships to other objects.** Do NOT modify capitalization. For example, if the user input mentions 'a fluffy white dog', this prompt should be exactly 'a fluffy white dog'. If the input is 'cat on a table', the prompt for the cat is 'cat' and the prompt for the table is 'a table' (or just 'table' depending on the exact wording around it)."
      }},
      // Add one entry per main object. STRICTLY follow this format.
    ]
  }}
}}

RULES FOR OBJECT SELECTION:
1. IDENTIFY MAIN OBJECTS ONLY:
   - Must be clearly described, physical, and significant in the scene.
   - Examples: cat → prop, couch → furniture, gothic library → room.

2. A **room object is always required**. If the scene description implies an outdoor setting without a defined room, you may need to infer a generic 'outdoor_space' or similar as the 'room' type object, and its prompt would be the relevant part of the user's description for that space.

3. STRICTLY EXCLUDE:
   - Lights, shadows, ambient/sunlight.
   - Fog, mist, atmosphere, dust.
   - Generic walls/floor/ceiling unless the room is the focused element and described as such.
   - Minor clutter (utensils, cushions, books) unless explicitly emphasized as a main object.

4. DEFAULT FIELD VALUES:
   - For 'name' and 'material', use inferred or standard values if not explicitly detailed for the object in the user's prompt. The 'prompt' field, however, must remain verbatim.

EXAMPLE SCENE AND REQUIRED OUTPUT:
Input: "A sleek black domestic cat lounges sitting on a beige couch"

Required Output (Demonstrating full structure, object inclusion, and verbatim prompts):
{{
  "scene": {{
    "objects": [
      {{
        "id": "1",
        "name": "black_cat",
        "type": "prop",
        "material": "fur",
        "prompt": "a sleek black domestic cat"
      }},
      {{
        "id": "2",
        "name": "beige_couch",
        "type": "furniture",
        "material": "fabric",
        "prompt": "a beige couch"
      }},
       {{
        "id": "3",
        "name": "living_room",
        "type": "room",
        "material": "walls",
        "prompt": "a cozy living room"
      }}
    ]
  }}
}}

STRICT ADHERENCE TO THIS FORMAT AND OBJECT INCLUSION IS ESSENTIAL FOR SUCCESSFUL RENDERING. Ensure all main physical objects described and the required room object are included. The 'prompt' field must be the exact, verbatim text from the input that *identifies or describes* that specific object, not its relationship to others.
"""
        self.user_prompt = "User: {user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = initialize_model(model_name, temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=InitialDecomposition)
        self.chain = self.prompt | self.model | self.parser

    def decompose(self, user_input: str) -> InitialDecompositionOutput:
        try:
            logger.info(f"Decomposing input: {user_input}")
            result: InitialDecomposition = self.chain.invoke({"user_input": user_input})
            logger.info(f"Decomposition result: {result}")

            output = InitialDecompositionOutput(
                decomposition=result, original_user_prompt=user_input
            )

            # Not relying on the llm to provide unique id for every object
            for obj_dict in output.decomposition.scene.objects:
                obj_dict.id = (
                    f"{obj_dict.name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}"
                )
            logger.info(f"Initial decomposition: {output}")
            return output
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            raise


class FinalDecomposerToolInput(BaseModel):
    improved_decomposition: dict = Field(
        description="Initial decomposition of user's request with enhaced prompts and original user prompt."
    )


@beartype
class FinalDecomposer:
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.system_prompt = """
You are a Spatial Layout and Scene Enrichment AI for a 3D rendering engine.

YOUR ROLE:
Given an initial user's structured scene decomposition JSON (which may have already had its object prompts enhanced), your task is to enrich it into a full 3D scene layout based on the inital user's prompt. This includes:
- Object positions, rotations, and scales.
- Classification of each object from the input as either 'primitive' or 'dynamic'.
- Generating unique IDs for all objects and lights.
- A suitable lighting setup (1-2 lights like sun, point, spot, or area).
- One skybox configuration (gradient, sun, or cubed).

INPUTS:
You will receive a JSON object (`improved_decomposition`) containing a list of pre-identified scene objects. Each object in `improved_decomposition` ALREADY HAS A UNIQUE `id` FIELD, a `name`, and a `prompt`. You will also receive the `original_user_prompt`.

YOUR CRITICAL TASK for objects from `improved_decomposition`:
- For each object provided in the `improved_decomposition.scene.objects` list:

  3. Determine its `type` ("dynamic" or "primitive") and `shape` (if primitive) based on its nature and prompt.
  4. Infer `position`, `rotation`, and `scale`.
  5. Set `path` to null for now.

YOUR TASK for NEW elements YOU create (lights, skybox components if they need IDs, additional primitives like ground planes):
- For any new elements you add to the scene (like lights, or perhaps a default ground plane if no 'room' was specified):
  1. **Generate a NEW, unique string `id` for each of these new elements.** (e.g., "directional_light_abc", "ground_plane_xyz").
  2. Populate all other required fields for these elements (position, color, intensity, type, etc.).

CRITICAL RULES:
---------------
1.  **PRESERVE PROMPTS:** You MUST use the `prompt` field for each object exactly as provided in the input JSON. Do NOT alter these prompt strings.
1.  **PRESERVE IDS:** You MUST use the `id` field for each object exactly as provided in the input JSON. Do NOT change or regenerate this ID.
2.  **PRESERVE NAMS:** You MUST use the `name` field for each object exactly as provided in the input JSON. Do NOT alter these name strings.
2.  **OBJECT TRANSFORMS:**
    -   `id`: Generate a unique string ID for each object (e.g., `objectName_randomSuffix`).
    -   `name`: Use the name from the input object.
    -   `position`: Infer sensible 3D coordinates. Place the 'room' object (if any from input type) generally around the origin (e.g., scale it up and center it). Place other objects relative to each other or the room, avoiding obvious overlaps. Default to near origin if no other context.
    -   `rotation`: Default to `{{"x": 0, "y": 0, "z": 0}}` unless orientation is clearly implied by the object's nature or its prompt.
    -   `scale`: Default to `{{"x": 1, "y": 1, "z": 1}}`. Adjust for objects that are typically very large (like a 'room' primitive) or if scale is implied.
3.  **OBJECT TYPE & SHAPE:**
    -   For each object from the input:
        -   If the input object's `type` (e.g., 'prop', 'furniture', 'mesh') suggests a complex, organic, or detailed model, set its `type` in the output to `"dynamic"`. `shape` should be `null`, `path` should be `null`.
        -   If the input object's `type` is 'room', or it describes a very simple geometric form (e.g., 'a large cube', 'a sphere'), set its `type` in the output to `"primitive"`. Assign an appropriate `shape` (e.g., "cube" for a room, "sphere", "plane"). `path` should be `null`.
4.  **LIGHTING:**
    -   Include 1-2 lights. Choose types (`directional`, `point`, `spot`, `area`) that logically match the overall scene description implied by the object prompts.
    -   `id`: Generate a unique string ID for each light.
    -   Provide all necessary fields for the chosen light type as per the schema (position, rotation, scale, color, intensity, indirect_multiplier, range, mode, shadow_type, etc.). Use reasonable defaults if not inferable. For example, a "sunny day" might imply a `directional` light.
5.  **SKYBOX:**
    -   Choose one skybox type: `gradient`, `sun`, or `cubed`.
    -   Provide all fields for the chosen skybox type with valid default values to set a basic mood. (e.g., `SunSkybox` for outdoor scenes, `GradientSkybox` or `CubedSkybox` for indoors).
6.  **ADHERE TO SCHEMA:** The entire output MUST be a single JSON object strictly conforming to the target Pydantic `Scene` model and its sub-models (`SceneObject`, `Light` variants, `Skybox` variants).

OUTPUT FORMAT (Return ONLY the JSON, ensure it matches the Pydantic models below):
{{
  "skybox": {{
    "type": "sun",
    "top_color": {{ "r": 0.2, "g": 0.4, "b": 0.8, "a": 1.0 }},
    "top_exponent": 1.0,
    "horizon_color": {{ "r": 0.6, "g": 0.7, "b": 0.8, "a": 1.0 }},
    "bottom_color": {{ "r": 0.8, "g": 0.8, "b": 0.7, "a": 1.0 }},
    "bottom_exponent": 1.0,
    "sky_intensity": 1.0,
    "sun_color": {{ "r": 1.0, "g": 0.95, "b": 0.85, "a": 1.0 }},
    "sun_intensity": 1.5,
    "sun_alpha": 0.8,
    "sun_beta": 0.6,
    "sun_vector": {{ "x": 0.3, "y": -0.7, "z": 0.2, "w": 0.0 }}
  }},
  "lights": [
    {{
      "id": "directional_light_01",
      "type": "directional",
      "position": {{ "x": 0, "y": 10, "z": 0 }},
      "rotation": {{ "x": 50, "y": -30, "z": 0 }},
      "scale": {{ "x": 1, "y": 1, "z": 1 }},
      "color": {{ "r": 1.0, "g": 0.95, "b": 0.9, "a": 1.0 }},
      "intensity": 1.0,
      "indirect_multiplier": 1.0,
      "mode": "realtime",
      "shadow_type": "soft_shadows"
    }},
    {{
      "id": "point_light_01",
      "type": "point",
      "position": {{ "x": -2, "y": 1.5, "z": 1 }},
      "rotation": {{ "x": 0, "y": 0, "z": 0 }},
      "scale": {{ "x": 1, "y": 1, "z": 1 }},
      "color": {{ "r": 1.0, "g": 0.8, "b": 0.6, "a": 1.0 }},
      "intensity": 0.8,
      "indirect_multiplier": 1.0,
      "range": 10.0,
      "mode": "mixed",
      "shadow_type": "hard_shadows"
    }}
  ],
  "objects": [
    {{
      "id": "black_cat_xyz123",
      "name": "black_cat",
      "type": "dynamic",
      "position": {{ "x": 0.5, "y": 0.6, "z": 0.1 }},
      "rotation": {{ "x": 0, "y": 15, "z": 0 }},
      "scale": {{ "x": 0.3, "y": 0.3, "z": 0.3 }},
      "path": null,
      "shape": null,
      "prompt": "a sleek black domestic cat"
    }},
    {{
      "id": "beige_couch_abc789",
      "name": "beige_couch",
      "type": "dynamic",
      "position": {{ "x": 0, "y": 0, "z": 0 }},
      "rotation": {{ "x": 0, "y": 0, "z": 0 }},
      "scale": {{ "x": 1.5, "y": 1.0, "z": 1.0 }},
      "path": null,
      "shape": null,
      "prompt": "a beige couch"
    }},
    {{
      "id": "living_room_wall_def456",
      "name": "living_room",
      "type": "primitive",
      "shape": "cube",
      "position": {{ "x": 0, "y": 1.5, "z": 0 }},
      "rotation": {{ "x": 0, "y": 0, "z": 0 }},
      "scale": {{ "x": 10, "y": 3, "z": 10 }},
      "path": null,
      "prompt": "a cozy living room"
    }}
  ]
}}
"""
        self.user_prompt = """
        Original User Prompt:
        {original_user_prompt}

        Decomposed Objects with IDs and Improved Prompts (Preserve these IDs for these objects):
        {improved_decomposition}

        Based on ALL the above information, generate the full scene JSON.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = initialize_model(model_name, temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=Scene)
        self.chain = self.prompt | self.model | self.parser

    def decompose(
        self,
        improved_decomposition: dict,
    ) -> FinalDecompositionOutput:
        try:
            validated_data = ImprovedDecompositionOutput(**improved_decomposition)
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for final_decomposer payload: {e}"
            )
            raise ValueError(
                f"Invalid payload structure for final_decomposer tool: {e}"
            )
        try:
            logger.info(
                f"Final decomposition with input: original_prompt='{validated_data.original_user_prompt}', improved_decomposition: {validated_data.decomposition.scene}."
            )

            result: Scene = self.chain.invoke(
                {
                    "original_user_prompt": validated_data.original_user_prompt,
                    "improved_decomposition": validated_data.decomposition.scene,
                }
            )
            logger.info(f"Decomposition result: {result}")

            return FinalDecompositionOutput(
                action="scene_generation",
                message="Scene description has been successfully generated.",
                final_scene_json=result,
            )

        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            raise


if __name__ == "__main__":
    # decomposer = InitialDecomposer()
    # superprompt = "A plush, cream-colored couch with a low back and rolled arms sits against a wall in a cozy living room. A sleek, gray cat with bright green eyes is curled up in the center of the couch, its fur fluffed out slightly as it sleeps, surrounded by a few scattered cushions and a worn throw blanket in a soft blue pattern."
    # output = decomposer.decompose(superprompt)
    # print(json.dumps(output, indent=2))

    decomposer = FinalDecomposer("llama3.1")
    superprompt = {
        "decomposition": {
            "scene": {
                "objects": [
                    {
                        "id": "1",
                        "name": "black_cat",
                        "type": "prop",
                        "material": "fur",
                        "prompt": "a sleek black domestic cat",
                    },
                    {
                        "id": "2",
                        "name": "beige_couch",
                        "type": "furniture",
                        "material": "fabric",
                        "prompt": "a beige couch",
                    },
                    {
                        "id": "3",
                        "name": "living_room",
                        "type": "room",
                        "material": "walls",
                        "prompt": "a cozy living room",
                    },
                ]
            }
        },
        "original_user_prompt": "A sleek black domestic cat lounges sitting on a beige couch in a cozy living room",
    }
    output = decomposer.decompose(superprompt)
    print(json.dumps(output, indent=2))
