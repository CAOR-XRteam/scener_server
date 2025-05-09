from loguru import logger
from colorama import Fore
from pydantic import BaseModel, Field
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import tool
import json


class DecomposeToolInput(BaseModel):
    prompt: str = Field(
        description="The improved user's scene description prompt to be decomposed."
    )


@beartype
class Decomposer:
    def __init__(self, temperature: float = 0.0):
        self.system_prompt = """
You are a highly specialized and precise Scene Decomposer for a 3D rendering workflow. Your single purpose is to accurately convert a scene description string into structured JSON according to strict rules.

YOUR CRITICAL TASK:
- First, decompose the scene description into several distinct elements, knowing that a scene should contain at least a room element.
- Convert the given scene description string into a valid JSON object.
- The decomposition MUST focus ONLY on the main, distinct objects of the scene DO NOT decompose every minor detail into its own object.
- Be sure that the number of elements are small enough (1-5), and that they represent the main elements of the scene.
- Place these element in the scene by specifying their position and rotation approximately to approach at maximum the scene description.

OUTPUT FORMAT:
- Return ONLY the JSON object.
- NO explanations, NO preambles, NO markdown formatting (like ```json), NO conversation.
- ABSOLUTELY ZERO TEXT before or after the JSON object.

JSON STRUCTURE (STRICT AND REQUIRED):
{{
  "scene": {{
    "objects": [
      {{
        "name": "concise_descriptive_object_name",  // Infer a clear, short name (e.g., 'black_cat', 'wooden_table')
        "type": "object_category",  // Use types like: 'mesh', 'furniture', 'prop', 'room' (if room is primary focus)
        "position": {{"x":0,"y":0,"z":0}}, // Use reasonable default or inferred position
        "rotation": {{"x":0,"y":0,"z":0}}, // Use reasonable default or inferred rotation
        "scale": {{"x":1,"y":1,"z":1}},    // Use reasonable default or inferred scale
        "material": "primary_material_name", // Describe the main material (e.g., 'polished_wood', 'glossy_fur', 'ceramic')
        "prompt": "Provide a detailed visual description of this object only without his contexte on the scene or his link with the other objects, with no environmental context. The description must mention that the object is placed on a 'white and empty background' and is 'completely detached' from its surroundings. If the object type is 'room', include the camera setting: 'room view from the outside with a distant 3/4 top-down perspective'. For all other types, use: 'front camera view'. we only want a good global illumination of the object."
      }},
      // Include one dictionary entry here for EACH identified main object.
      // FOLLOW THE STRUCTURE EXACTLY for all objects.
    ]
  }}
}}

STRICT RULES FOR SELECTING AND DESCRIBING OBJECTS:
1.  IDENTIFY MAIN SUBJECTS: Read the input and identify the primary, distinct physical entities that are clearly described and seem intended as key components of the scene. These are your ONLY candidates for 'object' entries.
    * Example Main Subjects: A specific animal (like a cat), a piece of furniture (a couch), a defined room (a house).
2.  **CRITICAL EXCLUSIONS - NEVER CREATE SEPARATE OBJECT ENTRIES FOR:**
    * **Lighting:** Do NOT create objects with type 'light' or names like 'ambient_light', 'sunlight', 'shadows'.
    * **Atmosphere/Effects:** Do NOT create objects for 'mist', 'fog', 'dust', 'haze', etc.
    * **General Environment:** Do NOT create objects for generic 'walls', 'floor', 'ceiling', or 'room' UNLESS the room itself is the main subject being described in detail (e.g., "a detailed gothic library room").
    * **Minor Details:** Do NOT create separate objects for very small or less significant items unless they are specifically highlighted (e.g., don't decompose a 'table_setting' into spoon, fork, knife unless the user specifically focuses on each utensil).
3.  CREATE OBJECT ENTRIES: For *each* main subject identified in Rule 1, create *one* complete dictionary entry in the 'objects' list following the JSON STRUCTURE EXACTLY.
4.  GENERATE PROMPTS: For each object entry's 'prompt' field:
    * Write a detailed description focusing *only* on the visual appearance of that specific object, drawing information from the input text.
    * Always precise in the prompt that the object should not have any background and be viewed from outside with a distance 3/4 top-down perspective, entire object must be entirely visible.*
    * Example final prompt string: `"A cat viewed ENTIRELY from outside a distance 3/4 top-down perspective, with no background. The cat is large, sleek black cat with glossy fur sitting calmly. "`
5.  USE DEFAULTS: Use reasonable default values for position, rotation, and scale unless the text explicitly provides spatial information relevant to that object.
6.  MATERIAL: Infer the primary material if described.

EXAMPLE
for the input prompt 'A sleek black domestic cat lounges on a beige couch in a cozy living room', there are 3 main objects: the cat, the couch and the room.
So you may describe the cat physical aspect only, the couch physical aspect only and the roo physical only in their respective json 'prompt' part.

STRICT ADHERENCE TO THESE RULES IS ESSENTIAL FOR SUCCESSFUL RENDERING. DOUBLE-CHECK YOUR OUTPUT AGAINST ALL RULES, ESPECIALLY THE EXCLUSIONS AND THE MANDATORY PROMPT ENDING FOR EVERY OBJECT.
"""
        self.user_prompt = "User: {improved_user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = OllamaLLM(model="gemma3:1b", temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=None)
        self.chain = self.prompt | self.model | self.parser

        # logger.info(f"Initialized with model: {model_name}")

    def decompose(self, improved_user_input: str) -> dict:
        try:
            logger.info(f"Decomposing input: {improved_user_input}")
            result: dict = self.chain.invoke(
                {"improved_user_input": improved_user_input}
            )
            logger.info(f"Decomposition result: {result}")
            return result
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            raise


@tool(args_schema=DecomposeToolInput)
def decomposer(prompt: str) -> dict:
    """Decomposes a user's scene description prompt into manageable elements for 3D scene creation."""
    logger.info(f"Using tool {Fore.GREEN}{'decomposer'}{Fore.RESET}")
    tool = Decomposer()
    output = tool.decompose(prompt)
    return output


if __name__ == "__main__":
    tool = Decomposer()
    superprompt = "A plush, cream-colored couch with a low back and rolled arms sits against a wall in a cozy living room. A sleek, gray cat with bright green eyes is curled up in the center of the couch, its fur fluffed out slightly as it sleeps, surrounded by a few scattered cushions and a worn throw blanket in a soft blue pattern."
    output = tool.decompose(superprompt)
    print(json.dumps(output, indent=2))
