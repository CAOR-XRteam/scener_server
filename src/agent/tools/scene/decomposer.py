import uuid

from agent.llm.creation import initialize_model
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from lib import logger
from pydantic import BaseModel, Field, ValidationError
from sdk.scene import Scene, FinalDecompositionOutput


class DecomposedObject(BaseModel):
    id: str
    name: str
    prompt: str


class DecompositionData(BaseModel):
    objects: list[DecomposedObject]


class Decomposition(BaseModel):
    scene: DecompositionData


class DecompositionOutput(BaseModel):
    scene_data: Decomposition
    original_user_prompt: str


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
   - For 'name', use inferred or standard value if not explicitly detailed for the object in the user's prompt. The 'prompt' field, however, must remain verbatim.

EXAMPLE SCENE AND REQUIRED OUTPUT:
Input: "A sleek black domestic cat lounges sitting on a beige couch"

Required Output (Demonstrating full structure, object inclusion, and verbatim prompts):
{{
  "scene": {{
    "objects": [
      {{
        "id": "1",
        "name": "black_cat",
        "prompt": "a sleek black domestic cat"
      }},
      {{
        "id": "2",
        "name": "beige_couch",
        "prompt": "a beige couch"
      }},
       {{
        "id": "3",
        "name": "living_room",
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
        self.parser = JsonOutputParser(pydantic_object=Decomposition)
        self.chain = self.prompt | self.model | self.parser

    def decompose(self, user_input: str) -> dict:
        try:
            logger.info(f"Decomposing input: {user_input}")
            result: Decomposition = self.chain.invoke({"user_input": user_input})
            logger.info(f"Decomposition result: {result}")

            output = DecompositionOutput(
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
You are a highly specialized AI that generates a complete 3D scene in a strict hierarchical JSON format. Your ONLY job is to create this JSON structure based on a list of objects and a scene description.
SCHEMA REFERENCE - YOU MUST FOLLOW THIS STRICTLY

The final output is a single JSON object with three top-level keys: name, skybox, and graph.

**1. "graph" (The Scene Hierarchy):**

    - The graph is a list of SceneObject nodes, which are the top-level objects in the scene.

    - Each SceneObject represents a container and MUST have these fields: id, position, rotation, scale, components, and children.

    - The children field is a list of other SceneObject nodes; it is always present in the structure, and if a SceneObject logically has no children, it must be an empty list [].

    - You must infer the relationships between objects based on the scene description. For example, if a lamp is inside a box, the lamp must be a child of the box.

**2. "components" (Defining what a SceneObject is):**

    - The components list is the most important part. Every item in it MUST have a component_type field.

        IMPORTANT: component_type MUST be one of "primitive", "dynamic", or "light".

        - If component_type is "primitive":

            The component object MUST also contain:

                shape: One of "cube", "sphere", "plane", "quad", "cylinder", "capsule".

                color: An RGBA color object.

        - If component_type is "dynamic":

            The component object MUST also contain:

                id: The name of the 3D model asset to load (e.g., "desk", "chair").

        - If component_type is "light":

            The component object MUST also contain a type field: "directional", "point", "spot", or "area".

            All light components MUST have: color, intensity, indirect_multiplier.

            For a Directional Light, you MUST also include: mode and shadow_type.

            For a Point Light, you MUST also include: range, mode, and shadow_type.

            For a Spot Light, you MUST also include: range, spot_angle, mode, and shadow_type.

            For an Area Light, you MUST also include: range, shape ("rectangle" or "disk"), and width/height (for rectangle) or radius (for disk).

**3. "skybox" (Global Scene Sky):**

    You MUST provide one skybox object.

    - If type is "sun", you MUST include all its fields: type, top_color, top_exponent, horizon_color, bottom_color, bottom_exponent, sun_color, sky_intensity, sun_intensity, sun_alpha, sun_beta, sun_vector (Unity Vector4 format with 'x', 'y', 'z' and 'w' fields).

    - If type is "gradient", you MUST include all its fields: color1, color2, up_vector (Unity Vector4 format with 'x', 'y', 'z' and 'w' fields), intensity, exponent.

    - If type is "cubed", you MUST include all its fields: tint_color, exposure, rotation, cube_map.

**IMPORTANT: Vectors (position, rotation, scale) have x, y, z fields.**
**IMPORTANT: Colors (color, top_color, etc.) are RGBA format (include 'r', 'g', 'b', and 'a' components).**

**EXAMPLE OF FINAL JSON OUTPUT (Use as a structural guide):**

"{{\"name\":\"A Dark Study Room\",\"skybox\":{{\"type\":\"gradient\",\"color1\":{{\"r\":0.1,\"g\":0.1,\"b\":0.2,\"a\":1}},\"color2\":{{\"r\":0.05,\"g\":0.05,\"b\":0.1,\"a\":1}},\"up_vector\":{{\"x\":0,\"y\":1,\"z\":0,\"w\":0}},\"intensity\":0.5,\"exponent\":1.0}},\"graph\":[{{\"id\":\"room_container\",\"position\":{{\"x\":0,\"y\":1.5,\"z\":0}},\"rotation\":{{\"x\":0,\"y\":0,\"z\":0}},\"scale\":{{\"x\":10,\"y\":3,\"z\":10}},\"components\":[{{\"component_type\":\"primitive\",\"shape\":\"cube\",\"color\":{{\"r\":0.2,\"g\":0.2,\"b\":0.25,\"a\":1]()_]()_

**CRITICAL RULES:**

    - JSON ONLY: Your entire output must be a single, valid JSON object. No extra text, no markdown, no explanations.

    - HIERARCHY IS KEY: Structure the objects logically using the children list. For example, a lamp should be a child of the room or the desk it sits on. Use local positions for children.

    - STRICT SCHEMA: Adhere strictly to the fields and values listed in the SCHEMA REFERENCE above. Every SceneObject must have a components list, even if it's empty.
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
    ) -> dict:
        try:
            validated_data = DecompositionOutput(**improved_decomposition)
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
        "scene_data": {
            "scene": {
                "objects": [
                    {
                        "id": "1",
                        "name": "cup_of_coffee",
                        "prompt": "a cup of coffee",
                    },
                    {
                        "id": "2",
                        "name": "wooden_table",
                        "prompt": "a wooden table",
                    },
                    {
                        "id": "3",
                        "name": "sunlit_kitchen",
                        "prompt": "a sunlit kitchen",
                    },
                ]
            }
        },
        "original_user_prompt": "A cup of coffee on a wooden table in a sunlit kitchen",
    }
    output = decomposer.decompose(superprompt)
    print(output)
