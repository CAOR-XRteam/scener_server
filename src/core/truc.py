from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=10000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def main():
    # Prompt
    template = """Question: {question} be polite and elegant and sober. speak only essential thing"""
    prompt = ChatPromptTemplate.from_template(template)

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Model
    model = OllamaLLM(model="gemma3:4b")

    # Chain
    chain = prompt | model | memory

    # Chat loop
    print("Chat with gemma3:4b (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        print("AI: ", end="", flush=True)
        for chunk in chain.stream({"question": user_input}):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    main()
