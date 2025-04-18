import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Decomposer:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        self.system_prompt = """You are a 3D scene generation assistant. You will receive a prompt describing a 3D scene that needs to be generated.

Your task is to break this prompt down into manageable elements that can be processed for 3D scene creation.

How to proceed:
1. **Check the Library First**: Before decomposing the scene, check the available elements in the library for existing objects, meshes, materials, and textures that closely match the user's input.
2. **Decompose Unavailable Elements**: If no matching elements are found, decompose the prompt into:
   - Object types (room, mesh, furniture)
   - Position, rotation, scale
   - Materials and textures (wood, stone, etc.)
   - Descriptive prompt per object

Output format:
Respond in JSON:
{{
  "scene": {{
    "objects": [
      {{
        "name": "theatre_room",
        "type": "room",
        "position": {{"x":0,"y":0,"z":0}},
        "rotation": {{"x":0,"y":0,"z":0}},
        "scale": {{"x":20,"y":10,"z":20}},
        "material": "traditional_wood_material",
        "prompt": "Generate an image of a squared traditional Japanese theatre room viewed from the outside..."
      }},
      {{
        "name": "hanging_lanterns",
        "type": "mesh",
        "position": {{"x":0,"y":5,"z":0}},
        "rotation": {{"x":0,"y":0,"z":0}},
        "scale": {{"x":0.5,"y":0.5,"z":0.5}},
        "material": "paper_lantern",
        "prompt": "Create a traditional paper lantern hanging in the air..."
      }}
    ]
  }}
}}"""
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
            result: dict = self.chain.invoke({"improved_user_input": improved_user_input})
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
