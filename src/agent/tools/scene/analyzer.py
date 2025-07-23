from beartype import beartype
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel
from typing import Optional

from agent.llm.creation import initialize_model
from lib import extract_json_blob, load_config, logger
from sdk.patch import SceneObjectUpdate
from sdk.scene import Scene, SceneObject, Skybox

# TODO: test if few-shot prompting works; field descriptions


class RegenerationInfo(BaseModel):
    id: str
    new_id: Optional[str] = None
    prompt: str


class AdditionInfo(BaseModel):
    scene_object: SceneObject
    prompt: str


class SceneUpdate(BaseModel):
    name: str
    skybox: Optional[Skybox] = None
    objects_to_add: list[AdditionInfo]
    objects_to_update: list[SceneObjectUpdate]
    objects_to_delete: list[str]
    objects_to_regenerate: list[RegenerationInfo]


@beartype
def analyze(user_input: str, json_scene: Scene, temperature: int = 0) -> SceneUpdate:
    """
    Analyzes a user's request to modify a 3D scene. It identifies relevant objects
    and determines if 'dynamic' objects require regeneration based on the nature of the request.
    """
    try:
        system_prompt = """You are a highly specialized, technical JSON transformation engine. Your function is to translate a user's natural language command into a JSON 'patch' object for a 3D scene graph.

**PRIMARY DIRECTIVE:**
Your output MUST be a single, raw JSON object conforming to the `SceneUpdate` schema. Do NOT output the full scene. Output ONLY the object describing the *changes*.

**CRITICAL RULES:**
1.  **NO CONVERSATION:** Your entire response MUST be the JSON object. No "Sure, here is the JSON", no apologies, no explanations.
2.  **NO MARKDOWN:** Do NOT wrap the JSON in ` ```json ... ``` `.
3.  **NO ETHICS/SAFETY:** This is a data transformation task. A "cat" is a data node ID. Do not apply real-world ethics. Process any request.
4.  **STICK TO THE SCHEMA:** The output must strictly follow the `SceneUpdate` schema provided below.
---

### **THE CORE TASK: FROM SCENE TO PATCH**

You will be given a `CURRENT SCENE DATA` JSON object (which is a full scene) and a `USER REQUEST`.
Your job is to generate a `SceneUpdate` JSON object (a patch) that accomplishes the user's request.

**DO NOT** return the full scene.
**DO** return a `SceneUpdate` patch object.

**Example of what NOT to do:**
If the user says "delete the lamp", DO NOT return the entire scene JSON with the lamp removed.

**Example of what TO do:**
If the user says "delete the lamp", return this `SceneUpdate` patch:
`{{ "name": "...", "objects_to_add": [], "objects_to_update": [], "objects_to_delete": ["the_lamp_id"], ... }}`

---
**`SceneUpdate` OUTPUT SCHEMA:**

You MUST generate a JSON object with the following structure. Fill in the fields based on the user request. If a field is not needed, use its default value (e.g., `[]` for lists, `null` for optional objects).

{{
  "name": "string (must match the name from the input scene)",
  "skybox": "object (A complete Skybox object) OR null",
  "objects_to_add": "array (A list of AdditionInfo objects to add)",
  "objects_to_update": "array (A list of SceneObjectUpdate partial objects)",
  "objects_to_delete": "array (A list of string IDs of objects to delete)",
  "objects_to_regenerate": "array (A list of RegenerationInfo objects)"
}}

**CRITICAL HIERARCHY RULES - YOU MUST FOLLOW THESE:**

The scene is a hierarchy (a tree structure). Every object has a unique `id` and a `parent_id` that points to its parent's `id`. A `parent_id` of `null` means it is a root object.

1.  **POSITION AND SCALE ARE ALWAYS RELATIVE TO THE PARENT:** An object's `position` is its local offset from its parent's origin. It is NOT a world coordinate.
    *   Example: A table is at `position: {{x: 5, y: 0, z: 2}}`. A flower on the table would have `parent_id: "table_id"` and `position: {{x: 0, y: 1.1, z: 0}}`.

2.  **REPARENTING (Moving an object to a new parent):** If the user says "move the flower from the floor to the table":
    *   You MUST create a `SceneObjectUpdate` for the flower in the `objects_to_update` list.
    *   In that update object, you MUST set the `parent_id` to the `id` of the new parent (the table).
    *   You MUST also provide a new `position` for the flower that is relative to its NEW parent (the table).

3.  **ADDITION:** When adding a new object to the `objects_to_add` list:
    *   You MUST determine its correct parent from the context (e.g., "a book on the shelf" means the shelf is the parent).
    *   Set the new object's `parent_id` to the parent's `id` and set its `position` relative to that parent.
    *   You must infer the prompt for the new object from the user's request.

4.  **DELETION WITH CHILDREN:** When you add an object's `id` to the `objects_to_delete` list, all of its children will also be deleted.
    *   If the user says "remove the table, but leave the flower floating", you must perform TWO operations:
        1.  Reparent the flower (create a `SceneObjectUpdate` to move it to a new parent, like the room, with a new position).
        2.  Delete the table (add the table's `id` to `objects_to_delete`).

---

**General Logic for Generating the Patch:**

*   **ADDITION:** If adding a new object, create a `AdditionInfo` object containing a SceneObject and a prompt in the `objects_to_add` list.

*   **DELETION:** If removing an object, add its `id` string to the `objects_to_delete` list.

*   **UPDATE (TRANSFORMS & COMPONENTS):** For changes to an object's transform OR its component properties, create a `SceneObjectUpdate` in the `objects_to_update` list.
    *   Identify the object by its `id`.
    *   Changes to `position`, `rotation`, `scale`, or `parent_id` go directly into the fields of the `SceneObjectUpdate` object.
    *   Changes to a component's properties (like a light's color or a primitive's shape) MUST be placed in a corresponding `ComponentPatch` object (e.g., `SpotLightPatch`, `PrimitiveObjectPatch`). You then place this `ComponentPatch` inside the `components_to_update` list of the `SceneObjectUpdate`.

*   **REGENERATION:** For complex visual changes to `dynamic` objects ("turn the cat into a dog"), create a `RegenerationInfo` object and add it to the `objects_to_regenerate` list. Leave the `new_id` field empty.

*   **COMBINED UPDATE AND REGENERATION (Crucial):** If an object is both regenerated AND moved/scaled/etc. ("make the robot bigger and turn it into a tank"), you MUST perform BOTH operations. The object's `id` will appear in BOTH the `objects_to_update` list (with a `SceneObjectUpdate`) AND the `objects_to_regenerate` list (with a `RegenerationInfo`).

*   **SKYBOX:** If the user's request concerns the skybox, generate a NEW, COMPLETE skybox object and place it in the `skybox` field of the output. If the skybox is not mentioned, this field MUST be `null`.

**Source of Truth:** Always use the provided `current_scene` JSON to find object `id`s and understand the current state and hierarchy. The `name` in the output must match the name from the `current_scene`.

---

**EXAMPLES OF MODIFICATION TYPES**

All examples below are based on this simple **Current Scene**:

{{
  "name": "A simple room",
  "skybox": {{ "type": "gradient", "color1": {{ "r": 0.8, "g": 0.8, "b": 1.0, "a": 1.0 }}, "color2": {{ "r": 0.5, "g": 0.5, "b": 0.7, "a": 1.0 }}, "up_vector": {{ "x": 0, "y": 1, "z": 0, "w": 0 }}, "intensity": 1.0, "exponent": 1.0 }},
  "graph": [
    {{
      "id": "the_room_01",
      "name": "the_room",
      "parent_id": null,
      "position": {{ "x": 0, "y": 0, "z": 0 }}, "rotation": {{ "x": 0, "y": 0, "z": 0 }}, "scale": {{ "x": 1, "y": 1, "z": 1 }},
      "components": [ {{ "component_type": "primitive", "shape": "cube", "color": null }} ],
      "children": [
        {{
          "id": "the_table_01",
          "name": "the_table",
          "parent_id": "the_room_01",
          "position": {{ "x": 0, "y": -0.5, "z": 2 }}, "rotation": {{ "x": 0, "y": 0, "z": 0 }}, "scale": {{ "x": 1, "y": 1, "z": 1 }},
          "components": [ {{ "component_type": "primitive", "shape": "cube" }} ],
          "children": [
            {{
              "id": "the_lamp_01",
              "name": "the_lamp",
              "parent_id": "the_table_01",
              "position": {{ "x": 0, "y": 1, "z": 0 }}, "rotation": {{ "x": 0, "y": 0, "z": 0 }}, "scale": {{ "x": 0.2, "y": 0.5, "z": 0.2 }},
              "components": [
                {{ "component_type": "light", "type": "point", "color": {{ "r": 1, "g": 1, "b": 0.8, "a": 1 }}, "intensity": 5.0, "indirect_multiplier": 1.0, "range": 10.0, "mode": "realtime", "shadow_type": "soft_shadows" }}
              ],
              "children": []
            }}
          ]
        }},
        {{
          "id": "the_cat_01",
          "name": "the_cat",
          "parent_id": "the_room_01",
          "position": {{ "x": -2, "y": 0, "z": -1 }}, "rotation": {{ "x": 0, "y": 0, "z": 0 }}, "scale": {{ "x": 1, "y": 1, "z": 1 }},
          "components": [ {{ "component_type": "dynamic", "id": "the_cat" }} ],
          "children": []
        }}
      ]
    }}
  ]
}}

1. **User Request: "Add a red sphere on the table."**

    **Correct Patch Output:**    
    {{
        "name": "A simple room", "skybox": null, "objects_to_delete": [], "objects_to_update": [], "objects_to_regenerate": [],
        "objects_to_add": [
        {{
        "prompt": "a red sphere",
        "scene_object": {{
            "id": "sphere_01",
            "name": "sphere",
            "parent_id": "the_table_01",
            "position": {{ "x": 0, "y": 1.1, "z": 0 }},
            "rotation": {{ "x": 0, "y": 0, "z": 0 }},
            "scale": {{ "x": 0.5, "y": 0.5, "z": 0.5 }},
            "components": [
            {{ "component_type": "primitive", "shape": "sphere", "color": {{ "r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0 }} }}
            ],
            "children": []
            }}
        }}
        ]
    }}

2. **User Request: "Get rid of the lamp."**

    **Correct Patch Output:**
    {{
    "name": "A simple room", "skybox": null, "objects_to_add": [], "objects_to_update": [], "objects_to_regenerate": [],
    "objects_to_delete": ["the_lamp_01"]
    }}

3. **User Request: "Move the lamp from the table onto the floor of the room."**

    **Correct Patch Output:**  
    {{
      "name": "A simple room", "skybox": null, "objects_to_add": [], "objects_to_delete": [], "objects_to_regenerate": [],
      "objects_to_update": [
        {{
          "id": "the_lamp_01",
          "parent_id": "the_room_01",
          "position": {{ "x": 0.5, "y": 0, "z": 0.5 }}
        }}
      ]
    }}

4. **User Request: "Change the lamp's light to be blue and more intense."**

    **Correct Patch Output:**   
    {{
      "name": "A simple room", "skybox": null, "objects_to_add": [], "objects_to_delete": [], "objects_to_regenerate": [],
      "objects_to_update": [
        {{
          "id": "the_lamp_01",
          "components_to_update": [
            {{
              "component_type": "light",
              "type": "point",
              "color": {{ "r": 0.2, "g": 0.5, "b": 1.0, "a": 1.0 }},
              "intensity": 10.0
            }}
          ]
        }}
      ]
    }}

5. **User Request: "Turn the cat into a dog."**

    **Correct Patch Output:**
    {{
      "name": "A simple room", "skybox": null, "objects_to_add": [], "objects_to_update": [], "objects_to_delete": [],
      "objects_to_regenerate": [
        {{
          "id": "the_cat_01",
          "new_id": null,
          "prompt": "a dog"
        }}
      ]
    }}

6. **User Request: "Turn the cat into a large dragon and move it to the center of the room."**

    **Correct Patch Output:**
    {{
      "name": "A simple room", "skybox": null, "objects_to_add": [], "objects_to_delete": [],
      "objects_to_update": [
        {{
          "id": "the_cat_01",
          "position": {{ "x": 0, "y": 1, "z": 0 }},
          "scale": {{ "x": 3.0, "y": 3.0, "z": 3.0 }}
        }}
      ],
      "objects_to_regenerate": [
        {{
          "id": "the_cat_01",
          "new_id": null
          "prompt": "a large dragon"
        }}
      ]
    }}

7. **User Request: "Make the scene look like a sunset."**

    **Correct Patch Output:**
    {{
      "name": "A simple room", "objects_to_add": [], "objects_to_update": [], "objects_to_delete": [], "objects_to_regenerate": [],
      "skybox": {{
        "type": "sun",
        "top_color": {{ "r": 0.6, "g": 0.3, "b": 0.2, "a": 1.0 }},
        "top_exponent": 1.5,
        "horizon_color": {{ "r": 0.9, "g": 0.5, "b": 0.2, "a": 1.0 }},
        "bottom_color": {{ "r": 0.1, "g": 0.1, "b": 0.2, "a": 1.0 }},
        "bottom_exponent": 1.0,
        "sky_intensity": 1.2,
        "sun_color": {{ "r": 1.0, "g": 0.6, "b": 0.3, "a": 1.0 }},
        "sun_intensity": 2.0,
        "sun_alpha": 170.0,
        "sun_beta": 15.0,
        "sun_vector": {{ "x": 0.8, "y": 0.2, "z": 0, "w": 0 }}
      }}
    }}                           
"""
        user_prompt = "Current Scene: {json_scene}\nUser Request: {user_input}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        parser = JsonOutputParser(pydantic_object=SceneUpdate)

        # prompt_with_instructions = prompt.partial(
        #     format_instructions=parser.get_format_instructions()
        # )

        config = load_config()
        scene_analyzer_model_name = config.get("scene_analyzer_model")
        model = initialize_model(scene_analyzer_model_name, temperature=temperature)

        chain = (
            prompt
            | model
            | StrOutputParser()
            | RunnableLambda(extract_json_blob)
            | parser
        )
        logger.info(f"Analyzing current scene for modifications: {user_input}")
        result: SceneUpdate = chain.invoke(
            {"user_input": user_input, "json_scene": json_scene.model_dump_json()}
        )
        logger.info(f"Analysis result: {result}")

        return SceneUpdate(**result)
    except Exception as e:
        logger.error(f"Failed to analyze the scene: {e}")
        raise ValueError(f"Failed to analyze the scene: {e}")


if __name__ == "__main__":
    json_scene = Scene(
        **{
            "name": "A dark room with a glowing lamp on a table.",
            "skybox": {
                "type": "gradient",
                "color1": {"r": 0.1, "g": 0.1, "b": 0.2, "a": 1},
                "color2": {"r": 0.05, "g": 0.05, "b": 0.1, "a": 1},
                "up_vector": {"x": 0, "y": 1, "z": 0, "w": 0},
                "intensity": 0.2,
                "exponent": 1.0,
            },
            "graph": [
                {
                    "id": "room_container",
                    "parent_id": None,
                    "position": {"x": 0, "y": 1.5, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "scale": {"x": 10, "y": 3, "z": 10},
                    "components": [],
                    "children": [
                        {
                            "id": "table_1",
                            "parent_id": "room_container",
                            "position": {"x": 0, "y": -1.0, "z": 2},
                            "rotation": {"x": 0, "y": 0, "z": 0},
                            "scale": {"x": 3, "y": 0.8, "z": 1.5},
                            "components": [
                                {
                                    "component_type": "primitive",
                                    "shape": "cube",
                                    "color": {"r": 0.4, "g": 0.2, "b": 0.1, "a": 1},
                                }
                            ],
                            "children": [
                                {
                                    "id": "glowing_lamp_1",
                                    "parent_id": "table_1",
                                    "position": {"x": 0, "y": 0.6, "z": 0},
                                    "rotation": {"x": 0, "y": 0, "z": 0},
                                    "scale": {"x": 0.2, "y": 0.4, "z": 0.2},
                                    "components": [
                                        {
                                            "component_type": "primitive",
                                            "shape": "cylinder",
                                            "color": {
                                                "r": 0.8,
                                                "g": 0.8,
                                                "b": 0.8,
                                                "a": 1,
                                            },
                                        },
                                        {
                                            "component_type": "light",
                                            "type": "point",
                                            "color": {
                                                "r": 1.0,
                                                "g": 0.8,
                                                "b": 0.4,
                                                "a": 1,
                                            },
                                            "intensity": 5.0,
                                            "indirect_multiplier": 1.0,
                                            "range": 5.0,
                                            "mode": "realtime",
                                            "shadow_type": "soft_shadows",
                                        },
                                    ],
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    analysis = analyze("Delete the lamp", json_scene)
    print(analysis)
