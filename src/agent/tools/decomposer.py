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
    def __init__(self, temperature: float = 0.5):
        self.system_prompt = """
You are a highly specialized and precise Scene Decomposer for a 3D rendering workflow. Your sole task is to accurately convert a scene description string into structured JSON, adhering to strict rules. The output must always generate **high-quality zero-shot prompts** for each object in the scene, following the format provided below.

YOUR CRITICAL TASK:
- Decompose the scene description into several distinct elements, ensuring at least one 'room' object.
- Convert the scene into a valid JSON object.
- Focus ONLY on **key physical elements**. **Do NOT extract minor details** or interpret the scene.
- Limit the total object count to 1–5, with always at least one 'room' type object.
- **The most critical part of the output is the 'prompt'** field, which must be a detailed, rich visual description of the object.

OUTPUT FORMAT:
- Output **ONLY** the JSON. **NO explanations**, **NO markdown**, **NO extra formatting**.

JSON STRUCTURE (STRICT AND REQUIRED):
{{
  "scene": {{
    "objects": [
      {{
        "name": "concise_descriptive_object_name",  // short, clear identifier (e.g., 'black_cat', 'wooden_table')
        "type": "object_category",  // one of: 'mesh', 'furniture', 'prop', 'room'
        "material": "primary_material_name",  // e.g., 'polished_wood', 'glossy_fur', 'ceramic'
        "prompt": "Detailed visual description of the object based on the input prompt but with more details. Must include: 'placed on a white and empty background, entire whole object, completely detached from the background'. Use 'front camera view' for all objects, and 'squared room view from the outside with a distant 3/4 top-down perspective' if the object type is 'room'. No mention of other objects or environmental context. Assume normalized object size and scale visually."
      }},
      // Add one entry per main object. STRICTLY follow this format.
    ]
  }}
}}

RULES FOR OBJECT SELECTION:
1. IDENTIFY MAIN OBJECTS ONLY:
   - Must be clearly described, physical, and significant in the scene.
   - Examples: cat → prop, couch → furniture, gothic library → room.
   - A **room object is always required**.

2. STRICTLY EXCLUDE:
   - Lights, shadows, ambient/sunlight.
   - Fog, mist, atmosphere, dust.
   - Generic walls/floor/ceiling unless the room is the focused element.
   - Minor clutter (utensils, cushions, books) unless explicitly emphasized.

3. **PROMPT FIELD**:
   - The prompt must provide a **rich, detailed description** of the object’s physical features.
   - Focus on the object’s **key design, material, and visible features**. Avoid mentioning relationships to other objects or placement in the scene.
   - The description should be **concise but full of visual details**, ensuring that the object is clearly distinguishable and detailed for rendering.
   - The prompt must include:
     - "Placed on a white and empty background."
     - "Completely detached from surroundings."
     - **Camera view** based on object type:
       - Use "front camera view" for non-room objects.
       - Use "squared room view from the outside with a distant 3/4 top-down perspective" for room objects.
   - Example prompt:
     "A traditional Japanese theater room with detailed wooden architecture, elevated tatami stage, red silk cushions, sliding shoji panels, and ornate golden carvings on the upper walls. The room is viewed from an isometric 3/4 top-down perspective, with an open cutaway style revealing the interior. The scene is well-lit with soft, global illumination. No people, no objects outside the room, placed on a white and empty background, completely detached from surroundings."

4. DEFAULT FIELD VALUES:
   - Use inferred or standard values for position, rotation, and scale unless explicitly specified.

EXAMPLE SCENE:
Input: "A sleek black domestic cat lounges on a beige couch in a cozy living room."

Generated Output (Objects):
- black_cat → type: prop
- beige_couch → type: furniture
- living_room → type: room

Each object will have its own **detailed visual prompt** with no mention of other objects or context, like so:
- "A sleek black domestic cat with glossy fur, lounging on a soft cushion. Front camera view, placed on a white and empty background, completely detached from its surroundings."
- "A plush beige couch with rounded arms and thick cushions, front camera view, placed on a white and empty background, completely detached from its surroundings."
- "A cozy living room with wooden furniture, a fireplace, and large windows, squared room view from the outside with a distant 3/4 top-down perspective, placed on a white and empty background, completely detached from surroundings."

STRICT ADHERENCE TO THESE RULES IS ESSENTIAL FOR SUCCESSFUL RENDERING. DO NOT ADD OR OMIT ANYTHING.
"""
        self.user_prompt = "User: {improved_user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = OllamaLLM(model="gemma3:4b", temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=None)
        self.chain = self.prompt | self.model | self.parser

        #logger.info(f"Initialized with model: {model_name}")

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
    superprompt = (
        "A plush, cream-colored couch with a low back and rolled arms sits against a wall in a cozy living room. A sleek, gray cat with bright green eyes is curled up in the center of the couch, its fur fluffed out slightly as it sleeps, surrounded by a few scattered cushions and a worn throw blanket in a soft blue pattern."
    )
    output = tool.decompose(superprompt)
    print(json.dumps(output, indent=2))
