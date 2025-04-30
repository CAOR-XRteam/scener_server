from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


def model_gemma3_4b():
    """Load and return the Ollama model."""
    return ChatOllama(model="gemma3:4b")

def model_qwen3_4b(tools):
    """Load and return the Ollama model."""
    llm = ChatOllama(model="qwen3:4b")
    memory = InMemorySaver()

    agent = create_react_agent(
        tools=tools,
        model=llm,
        checkpointer=memory
    )
    return agent
