from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from model.llm.improver import Improver


class Chatbot:
    def __init__(self):
        # Define the template for the prompt
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful 3D engine chatbot assistant. answer briefly to the question in one or two paragraphes and do not speak outside the question scope.",
            },
            {"role": "user", "content": ""},
        ]

        # Tools
        self.model = OllamaLLM(model="gemma3:4b", streaming=True)  # Enable streaming
        # self.llm_improver = Impover()

        # Agentic executor
        # self.tools = [search]
        # agent_executor = create_react_agent(model, tools, checkpointer=memory)

    def chat(self, user_input):
        """Send a prompt to the LLM and receive a structured response."""
        self.messages.append({"role": "user", "content": user_input})
        prompt = ChatPromptTemplate.from_messages(self.messages)
        chain = prompt | self.model

        # Use the LangChain chain to generate a response
        response = chain.stream({})

        # Stream and print the response
        result = ""
        for chunk in response:
            result += chunk
            print(chunk, end="", flush=True)  # Print each chunk in the stream
        print("\n")

        return result

    def run(self):
        while True:
            user_input = input("Chat: ")
            response = self.chat(user_input)


# Usage
if __name__ == "__main__":
    chat = Chatbot()
    chat.run()
