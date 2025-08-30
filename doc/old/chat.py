from ollama import chat


class Chat:
    def __init__(self):
        self.list_message = [
            {
                "role": "system",
                "content": 'You are a 3D engine chatbot assistant. Be concise, calm, and friendly. \
                By default, you act as a chatbot. If the user requests image generation, \
                run a prompt with "no background" added if explicitly requested. \
                If asked about your capabilities, mention you can generate an image and assist with chat. \
                Do not provide image examples. \
                When generating an image, return a JSON response: {"command": "generate_image", "prompt": "your_prompt"}.',
            },
        ]

    def prompt(self, user_input):
        """Send a prompt to the LLM and receive a structured response."""
        self.list_message.append({"role": "user", "content": user_input})
        response = chat("llama3.2", messages=self.list_message)
        self.list_message.append(
            {"role": "assistant", "content": response.message.content}
        )
        return response.message.content

    def run(self):
        while True:
            user_input = input("Chat: ")
            response = self.prompt(user_input)
            print(response + "\n")


# Usage
if __name__ == "__main__":
    chat = Chat()
    chat.run()
