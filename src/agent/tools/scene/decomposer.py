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

    def decompose(self, user_input: str) -> dict:
        try:
            logger.info(f"Decomposing input: {user_input}")
            result: InitialDecomposition = self.chain.invoke({"user_input": user_input})
            logger.info(f"Decomposition result: {result}")

            output = InitialDecompositionOutput(
                scene_data=result, original_user_prompt=user_input
            )

            # Not relying on the llm to provide unique id for every object
            for obj_dict in output.scene_data.scene.objects:
                obj_dict.id = (
                    f"{obj_dict.name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}"
                )
            logger.info(f"Initial decomposition: {output}")
            return output.model_dump()
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
You are a highly specialized AI that converts a list of scene objects into a complete 3D scene layout in a strict JSON format. Your ONLY job is to fill out the JSON structure according to the schemas and guidance provided below.

YOUR TASK IS TO FOLLOW THESE STEPS EXACTLY:

STEP 1: POPULATE THE "objects" ARRAY
- For each object from the `improved_decomposition` input, create a corresponding JSON object in the output `objects` array.
- You MUST copy these fields EXACTLY as they appear in the input: `id`, `name`, `prompt`. DO NOT CHANGE THEM.
- You will INFER and ADD the following new fields for each object:
  - `type`: Set to "dynamic" for complex items (cat, couch). Set to "primitive" for simple shapes (room, ground plane).
  - `shape`: If `type` is "primitive", set this to one of: "cube", "sphere", "capsule", "cylinder", "plane", "quad". Otherwise, set it to `null`.
  - `position`, `rotation`, `scale`: Infer reasonable 3D transform values to create a logical scene.
  - `path`: Always set to `null`.

STEP 2: POPULATE THE "lights" ARRAY
- Create a `lights` array. Add 1-2 lights to it.
- For each new light, you MUST generate a NEW, UNIQUE `id` (e.g., "directional_light_1").
- Choose a light `type` based on the guidance below and provide ALL of its required fields. DO NOT mix fields from different types.

--- LIGHTING GUIDANCE & CHOICES ---
- **Contextual Hints**:
  - For **outdoor scenes**, use one `"directional"` light to act as the sun.
  - For **indoor scenes**, use `"point"` lights for general illumination (like lightbulbs) or `"spot"` lights to highlight specific objects (like a desk lamp).
  - Use `"area"` lights for soft, realistic lighting from a surface like a window or a studio softbox.

- **Schema Definitions**:
  1. If `type` is `"directional"`:
     - Required fields: `id`, `position`, `rotation`, `scale`, `color`, `intensity`, `indirect_multiplier`, `mode` ("baked", "mixed", or "realtime"), `shadow_type` ("no_shadows", "hard_shadows", or "soft_shadows").
  2. If `type` is `"point"`:
     - Required fields: `id`, `position`, `rotation`, `scale`, `color`, `intensity`, `indirect_multiplier`, `range`, `mode`, `shadow_type`.
  3. If `type` is `"spot"`:
     - Required fields: `id`, `position`, `rotation`, `scale`, `color`, `intensity`, `indirect_multiplier`, `range`, `spot_angle`, `mode`, `shadow_type`.
--- END LIGHTING ---

STEP 3: POPULATE THE "skybox" OBJECT
- Create a single `skybox` object.
- Choose ONE `type` based on the guidance below and provide ALL of its required fields.

--- SKYBOX GUIDANCE & CHOICES ---
- **Contextual Hints**:
  - For **outdoor scenes** (e.g., "a field", "a sunny beach"), use the `"sun"` type.
  - For **indoor scenes** (e.g., "a cozy room", "a dark library"), `"gradient"` is a great choice for setting a general mood.
  - Use `"cubed"` for realistic reflections, especially in indoor or studio scenes.

- **Schema Definitions**:
  1. If `type` is `"sun"`:
     - Required fields: `type`, `top_color`, `top_exponent`, `horizon_color`, `bottom_color`, `bottom_exponent`, `sky_intensity`, `sun_color`, `sun_intensity`, `sun_alpha`, `sun_beta`, `sun_vector` (Unity Vector4 format with 'x', 'y', 'z' and 'w' fields).
  2. If `type` is `"gradient"`:
     - Required fields: `type`, `color1`, `color2`, `up_vector` (Unity Vector4 format with 'x', 'y', 'z' and 'w' fields), `intensity`, `exponent`.
  3. If `type` is `"cubed"`:
     - Required fields: `type`, `tint_color`, `exposure`, `rotation`, `cube_map` (provide a placeholder string like "default_cubemap").
--- END SKYBOX ---

**IMPORTANT**: Colors are of RGBA format (include 'r', 'g', 'b' and 'a' components).

CRITICAL RULES - DO NOT VIOLATE:
1.  **JSON ONLY**: Your entire output MUST be a single, valid JSON object.
2.  **NO EXTRA TEXT**: Do NOT include markdown ```json ```, explanations, or any text outside of the final JSON object. No thinking.
3.  **PRESERVE INPUT DATA**: `id`, `name`, and `prompt` for objects from the input list are sacred. Copy them verbatim.
4.  **GENERATE NEW IDs**: Any new elements you add (lights) MUST have a new, unique ID that you generate.
5.  **ADHERE TO THE SCHEMA**: Your output MUST strictly conform to the schemas described in Steps 1, 2, and 3. Do not add or omit fields for a chosen type.
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
                f"Final decomposition with input: original_prompt='{validated_data.original_user_prompt}', improved_decomposition: {validated_data.scene_data.scene}."
            )

            result: Scene = self.chain.invoke(
                {
                    "original_user_prompt": validated_data.original_user_prompt,
                    "improved_decomposition": validated_data.scene_data.model_dump(),
                }
            )
            logger.info(f"Decomposition result: {result}")

            return FinalDecompositionOutput(
                action="scene_generation",
                message="Scene description has been successfully generated.",
                final_scene_json=result,
            ).model_dump()

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
    print(output)
