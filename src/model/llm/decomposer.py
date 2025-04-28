import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Decomposer:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        self.system_prompt = """You are a specialized JSON formatting assistant for 3D scene descriptions.
Your SOLE TASK is to convert a given scene description string into a structured JSON object.

INPUT: You will receive a detailed text description of a 3D scene.

OUTPUT REQUIREMENTS:
1.  **JSON ONLY**: Your response MUST be a single, valid JSON object and NOTHING ELSE.
2.  **NO EXTRA TEXT**: Do NOT include explanations, apologies, greetings, comments, markdown formatting (like ```json ... ```), or any text before or after the JSON object.
3.  **EXACT STRUCTURE**: The JSON object MUST strictly adhere to the following structure:

{{
  "scene": {{
    "objects": [
      {{
        "name": "object_name_here",  // Infer a concise, descriptive name
        "type": "object_type_here",  // e.g., room, mesh, furniture, light
        "position": {{"x":0,"y":0,"z":0}}, // Default or inferred position
        "rotation": {{"x":0,"y":0,"z":0}}, // Default or inferred rotation
        "scale": {{"x":1,"y":1,"z":1}},    // Default or inferred scale (adjust defaults if necessary, e.g., for a room)
        "material": "material_name_here", // e.g., wood, glossy_red, metallic_silver
        "prompt": "Descriptive sub-prompt for rendering this specific object visually" // A detailed prompt snippet focused *only* on this object
      }},
      // Add more objects here following the same structure for every distinct element described in the input text.
    ]
  }}
}}

PROCESSING LOGIC:
-   Carefully read the input description.
-   Identify each distinct object, element, or area (e.g., floor, wall, table, lamp, character).
-   For each identified element, create a corresponding object entry in the JSON `objects` array.
-   Fill in the fields (`name`, `type`, `position`, `rotation`, `scale`, `material`, `prompt`) based on the description. Use reasonable defaults if details are missing, but prioritize information from the text.
-   The `prompt` field for each object should be a concise description focused *solely* on that object's appearance and characteristics as derived from the input text.

Example Input: "A large, empty warehouse room with concrete floors, brick walls, and a metal rolling door on the far wall. A single wooden crate sits in the center."

Example (Partial) Output Structure:
{{
  "scene": {{
    "objects": [
      {{
        "name": "warehouse_room_floor",
        "type": "mesh", // Or "floor" if you have specific types
        "position": {{"x":0,"y":0,"z":0}},
        "rotation": {{"x":0,"y":0,"z":0}},
        "scale": {{"x":20,"y":0.1,"z":20}}, // Example scale
        "material": "concrete",
        "prompt": "A wide expanse of smooth, grey concrete floor showing some wear."
      }},
      {{
        "name": "warehouse_wall_brick",
        "type": "mesh", // Or "wall"
        // ... position/rotation/scale for one wall
        "material": "red_brick",
        "prompt": "A tall wall made of aged red bricks with visible mortar lines."
      }},
      // ... other walls ...
      {{
        "name": "metal_rolling_door",
        "type": "mesh", // Or "door"
        // ... position/rotation/scale for the door on a specific wall
        "material": "corrugated_metal",
        "prompt": "A large, grey, corrugated metal rolling door, closed."
      }},
      {{
        "name": "wooden_crate",
        "type": "mesh", // Or "prop", "furniture"
        "position": {{"x":0,"y":0.5,"z":0}}, // Assuming center, slightly above floor
        "rotation": {{"x":0,"y":0,"z":0}},
        "scale": {{"x":1,"y":1,"z":1}},
        "material": "weathered_wood",
        "prompt": "A standard-sized wooden shipping crate, showing signs of wear, placed centrally."
      }}
    ]
  }}
}}
"""
        self.user_prompt = "User: {improved_user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = OllamaLLM(model=model_name, temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=None)
        self.chain = self.prompt | self.model | self.parser

        logger.info(f"Initialized with model: {model_name}")

    def decompose(self, improved_user_input: str) -> dict:
        try:
            logger.info(f"Decomposing input: {improved_user_input}")
            result: dict = self.chain.invoke(
                {"improved_user_input": improved_user_input}
            )
            return result
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            raise

        # prompt = f"User: {improved_user_input}"
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": prompt},
        # ]

        # return deserialize_from_str(
        #     chat_call(self.model_name, messages, logger), logger
        # )


if __name__ == "__main__":
    decomposer = Decomposer()
    enhanced_user_prompt = (
        "Generate a traditional Japanese theatre room with intricate wooden flooring, "
        "high wooden ceiling beams, elegant red and gold accents, and large silk curtains."
    )
    print(result=decomposer.decompose(enhanced_user_prompt))
