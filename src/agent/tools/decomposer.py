from ollama import chat


class Decomposer:
    def __init__(self):
        self.list_message = [
            {
                'role': 'system',
                'content': 'You are a 3D scene generation assistant. You will receive a prompt describing a 3D scene that needs to be generated. Your task is to break this prompt down into manageable elements that can be processed for 3D scene creation. Here is how you should approach the task: Check the Library First: Before decomposing the scene, check the available elements in the library for existing objects, meshes, materials, and textures that closely match the users input. This will help in reusing pre-existing assets to save time and maintain coherence.Decompose Unavailable Elements: If there are no matching elements in the library, you will need to decompose the prompt into smaller components for individual creation. This might include: Objects and their types (e.g., room, mesh, furniture) Position, rotation, and scale of objects Materials and textures (e.g., wood, stone, paper) Descriptive prompts to generate specific assets. Output Format: The response should be formatted as a JSON object, following this structure, where each object in the scene has its own attributes like name, type, position, rotation, scale, material, and a descriptive prompt. Example JSON output format: {"scene":{"objects":[{"name":"theatre_room","type":"room","position":{"x":0,"y":0,"z":0},"rotation":{"x":0,"y":0,"z":0},"scale":{"x":20,"y":10,"z":20},"material":"traditional_wood_material","prompt":"Generate an image of a squared traditional Japanese theatre room viewed from the outside at a 3/4 top-down perspective. The room has intricate wooden flooring, high wooden ceiling beams, elegant red and gold accents, large silk curtains, bamboo poles, folding screens, and a scenic backdrop of mountains and cherry blossoms. The essence of classical Japanese theatre, such as Kabuki or Noh, should be captured, with a serene and elegant atmosphere."},{"name":"hanging_lanterns","type":"mesh","position":{"x":0,"y":5,"z":0},"rotation":{"x":0,"y":0,"z":0},"scale":{"x":0.5,"y":0.5,"z":0.5},"material":"paper_lantern","prompt":"Create a traditional paper lantern hanging in the air. The lantern should have intricate patterns, soft glowing light, and an artistic design, blending historical authenticity with a mythical aesthetic. The lantern should have red and gold accents with a soft, glowing warmth emanating from within."}]}}'
            },
        ]

    def prompt(self, user_input):
        """Send the user input to the LLM and receive the enhanced response."""
        self.list_message.append({'role': 'user', 'content': user_input})
        response = chat('llama3.2', messages=self.list_message)
        self.list_message.append({'role': 'assistant', 'content': response.message.content})
        return response.message.content

    def run(self):
        """Interact with the user to continuously improve prompts."""
        print("Welcome to the prompt improver! Type your prompts below.")
        while True:
            user_input = input("Improver: ")
            if user_input.lower() == "exit":
                print("Exiting the improver. Goodbye!")
                break
            response = self.prompt(user_input)
            print(f"Improved Prompt: {response}\n")


# Usage
if __name__ == '__main__':
    improver = Improver()
    improver.run()
