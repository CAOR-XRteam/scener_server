from loguru import logger
from colorama import Fore
from pydantic import BaseModel, Field
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import tool


class DecomposeToolInput(BaseModel):
    prompt: str = Field(
        description="The improved user's scene description prompt to be decomposed."
    )

@beartype
class Decomposer:
    def __init__(self, temperature: float = 0.0):
        self.system_prompt = """You are a highly specialized and precise Scene Decomposer for a 3D rendering workflow. Your single purpose is to accurately convert a scene description string into structured JSON according to strict rules.

YOUR CRITICAL TASK:
- Convert the given scene description string into a valid JSON object.
- The decomposition MUST focus ONLY on the main, distinct renderable subjects or primary elements that the user wants images of. DO NOT decompose every minor detail into its own object.

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
        "prompt": "Detailed visual description of ONLY this object based on input, including relevant environmental context, followed by mandatory phrases." // A detailed prompt snippet for rendering *this specific object*.
      }},
      // Include one dictionary entry here for EACH identified main object.
      // FOLLOW THE STRUCTURE EXACTLY for all objects.
    ]
  }}
}}

STRICT RULES FOR SELECTING AND DESCRIBING OBJECTS:
1.  IDENTIFY MAIN SUBJECTS: Read the input and identify the primary, distinct physical entities or areas that are clearly described and seem intended as key components of the scene. These are your ONLY candidates for 'object' entries.
    * Example Main Subjects: A specific animal (like the cat), a piece of furniture (the table), a defined prop (the mug).
2.  **CRITICAL EXCLUSIONS - NEVER CREATE SEPARATE OBJECT ENTRIES FOR:**
    * **Lighting:** Do NOT create objects with type 'light' or names like 'ambient_light', 'sunlight', 'shadows'.
    * **Atmosphere/Effects:** Do NOT create objects for 'mist', 'fog', 'dust', 'haze', etc.
    * **General Environment:** Do NOT create objects for generic 'walls', 'floor', 'ceiling', or 'room' UNLESS the room itself is the main subject being described in detail (e.g., "a detailed gothic library room").
    * **Minor Details:** Do NOT create separate objects for very small or less significant items unless they are specifically highlighted (e.g., don't decompose a 'table_setting' into spoon, fork, knife unless the user specifically focuses on each utensil).
3.  INCORPORATE ENVIRONMENTAL DETAILS: Describe lighting effects, shadows, atmospheric conditions (like mist), and relevant background context *within the 'prompt' field* of the main object(s) that are visually affected by these elements. For example, describe "soft light on the cat's fur" in the cat's prompt.
4.  CREATE OBJECT ENTRIES: For *each* main subject identified in Rule 1, create *one* complete dictionary entry in the 'objects' list following the JSON STRUCTURE EXACTLY.
5.  GENERATE PROMPTS: For each object entry's 'prompt' field:
    * Write a detailed description focusing *only* on the visual appearance of that specific object, drawing information from the input text.
    * Include how environmental factors (lighting, mist) described in the input affect this object's look.
    * Always precise in the prompt that the object should not have any background and be viewed from outside with a distance 3/4 top-down perspective, entire object must be visible.*
    * Example final prompt string: `"A cat viewed ENTIRELY from outside a distance 3/4 top-down perspective, with no background. The cat is large, sleek black cat with glossy fur sitting calmly. Soft light casts subtle highlights on the fur."`
6.  USE DEFAULTS: Use reasonable default values for position, rotation, and scale unless the text explicitly provides spatial information relevant to that object.
7.  MATERIAL: Infer the primary material if described.

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
        "Generate a traditional Japanese theatre room with intricate wooden flooring, high wooden ceiling beams, elegant red and gold accents, and large silk curtains."
    )
    output = tool.decompose(superprompt)
    print(output)
