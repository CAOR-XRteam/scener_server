from ollama import chat
from huggingface_hub import InferenceClient
from PIL import Image


class Chat:
    def __init__(self):
        self.list_message = [
            {
                'role': 'system',
                'content': 'You are a 3D engine chatbot assistant. Be concise, calm, and friendly. \
                By default, you act as a chatbot. If the user requests image generation, \
                run a prompt with "no background" added if explicitly requested. \
                If asked about your capabilities, mention you can generate an image and assist with chat. \
                Do not provide image examples. \
                When generating an image, return a JSON response: {"command": "generate_image", "prompt": "your_prompt"}.',
            },
        ]

    def prompt(self, user_input):
        """Send a prompt to the LLM and receive a structured response."""
        self.list_message.append({'role': 'user', 'content': user_input})
        response = chat('llama3.2', messages=self.list_message)
        answer = response.message.content
        self.list_message.append({'role': 'assistant', 'content': answer})
        return answer

    def run(self):
        while True:
            user_input = input("Chat: ")
            prompt = self.chat_with_llm(user_input)
            print(prompt + '\n')


# Usage
if __name__ == '__main__':
    generator = LLM(hf_token="")
    generator.run()
