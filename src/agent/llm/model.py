"""
model.py

LLM models and Langchain agent configuration

Author: Artem
Created: 05-05-2025
Last Updated: 05-05-2025
"""

from beartype import beartype
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


def gemma3_4b():
    """Load and return the Ollama model."""
    return ChatOllama(model="gemma3:4b")


def qwen3_8b():
    """Load and return the Ollama model."""
    return ChatOllama(model="qwen3:8b", streaming=True)


@beartype
def initialize_agent(model: str, tools: list[BaseTool], base_prompt: str):
    """Initialize the agent with the specified tools and prompt."""
    llm = ChatOllama(model=model, streaming=True)
    memory = InMemorySaver()

    agent = create_react_agent(
        tools=tools, model=llm, prompt=base_prompt, checkpointer=memory
    )
    return agent
