from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from model.llm.improver import improver_tool


class Agent:
    def __init__(self):
        # Define the template for the prompt
        self.system_prompt = (
            "You are a helpful 3D engine chatbot assistant. "
            "Answer briefly in one or two paragraphs and do not go outside the question scope."
        )

        # Memory and Model
        self.memory = MemorySaver()
        self.model = OllamaLLM(model="gemma3:4b", streaming=True)
        self.tools = [improver_tool]
        self.agent = create_react_agent(self.model, self.tools, checkpointer=self.memory)

    def chat(self, user_input):
        """Send a prompt to the LLM and receive a structured response."""
        prompt = f"{self.system_prompt}\nUser: {user_input}\nAssistant:"
        response_stream = self.model.stream(prompt)

        # Print response in real-time
        response = ""
        for chunk in response_stream:
            response += chunk
            print(chunk, end='', flush=True)  # Print each chunk
        print("\n")  # Newline after completion

        return response

    def run(self):
        print("Type 'exit' to quit.")
        while True:
            user_input = input("Chat: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            self.chat(user_input)


# Usage
if __name__ == '__main__':
    print("-------------------")
    chat = Agent()
    chat.run()
