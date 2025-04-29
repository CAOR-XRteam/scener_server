import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Decomposer:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        self.system_prompt = """You are a specialized Scene Decomposer for 3D generation.

YOUR TASK:
- Convert a given scene description string into a valid structured JSON object.

OUTPUT FORMAT:
- Return ONLY the JSON object.
- NO explanations, NO preambles, NO markdown.
- ABSOLUTELY NO TEXT before or after the JSON.

JSON STRUCTURE (STRICT):
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


RULES:
- Identify every distinct object or area (e.g., walls, floors, furniture, lights).
- Create an `object` entry for each.
- Use reasonable default values if necessary, but prefer information from the text.
- The `prompt` field must focus ONLY on the visual description of that object.
- NO additional text, commentary, or formatting beyond the required JSON object.
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
