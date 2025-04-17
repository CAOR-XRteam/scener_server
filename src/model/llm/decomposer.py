from ollama import chat
import json
from beartype import beartype
import logging


@beartype
class Decomposer:
    @beartype
    def __init__(self, model_name: str = "llama3.2"):
        self.system_prompt = 'You are a 3D scene generation assistant. You will receive a prompt describing a 3D scene that needs to be generated. Your task is to break this prompt down into manageable elements that can be processed for 3D scene creation. Here is how you should approach the task: Check the Library First: Before decomposing the scene, check the available elements in the library for existing objects, meshes, materials, and textures that closely match the users input. This will help in reusing pre-existing assets to save time and maintain coherence. Decompose Unavailable Elements: If there are no matching elements in the library, you will need to decompose the prompt into smaller components for individual creation. This might include: Objects and their types (e.g., room, mesh, furniture) Position, rotation, and scale of objects Materials and textures (e.g., wood, stone, paper) Descriptive prompts to generate specific assets. Output Format: The response should be formatted as a JSON object, following this structure, where each object in the scene has its own attributes like name, type, position, rotation, scale, material, and a descriptive prompt. Example JSON output format: {"scene":{"objects":[{"name":"theatre_room","type":"room","position":{"x":0,"y":0,"z":0},"rotation":{"x":0,"y":0,"z":0},"scale":{"x":20,"y":10,"z":20},"material":"traditional_wood_material","prompt":"Generate an image of a squared traditional Japanese theatre room viewed from the outside at a 3/4 top-down perspective. The room has intricate wooden flooring, high wooden ceiling beams, elegant red and gold accents, large silk curtains, bamboo poles, folding screens, and a scenic backdrop of mountains and cherry blossoms. The essence of classical Japanese theatre, such as Kabuki or Noh, should be captured, with a serene and elegant atmosphere."},{"name":"hanging_lanterns","type":"mesh","position":{"x":0,"y":5,"z":0},"rotation":{"x":0,"y":0,"z":0},"scale":{"x":0.5,"y":0.5,"z":0.5},"material":"paper_lantern","prompt":"Create a traditional paper lantern hanging in the air. The lantern should have intricate patterns, soft glowing light, and an artistic design, blending historical authenticity with a mythical aesthetic. The lantern should have red and gold accents with a soft, glowing warmth emanating from within."}]}}'
        self.model_name = model_name
        logging.info(f"Decomposer initialized with model: {self.model_name}")

    @beartype
    def decompose(self, improved_user_input: str) -> dict:
        prompt = f"User: {improved_user_input}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = chat(self.model_name, messages)
        except Exception as e:
            logging.error(f"Error during Decomposer chat API call: {str(e)}")
            return {"error": f"Decomposer chat API call failed: {str(e)}"}
        try:
            decomposition = json.loads(response.message.content)
        except (json.JSONDecodeError, ValueError) as e:
            decomposition = {
                "error": f"Invalid JSON: {str(e)}",
                "raw_response": response.message.content if response else None,
            }
        logging.info(f"Decomposer chat API call successful: {decomposition}")
        return decomposition


if __name__ == "__main__":
    decomposer = Decomposer()
    enhanced_user_prompt = (
        "Generate a traditional Japanese theatre room with intricate wooden flooring, "
        "high wooden ceiling beams, elegant red and gold accents, and large silk curtains."
    )
    try:
        result = decomposer.decompose(enhanced_user_prompt)
        print(result)
    except Exception as e:
        print(f"Decomposition failed: {str(e)}")
