from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


def gemma3_4b():
    """Load and return the Ollama model."""
    return ChatOllama(model="gemma3:4b")

def qwen3_4b(tools):
    """Load and return the Ollama model."""
    llm = ChatOllama(model="qwen3:4b", temperature=0.5)
    memory = InMemorySaver()

    agent = create_react_agent(
        tools=tools,
        model=llm,
        checkpointer=memory
    )
    return agent
