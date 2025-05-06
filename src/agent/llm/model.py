"""
model.py

LLM models and Langchain agent configuration

Author: Artem
Created: 05-05-2025
Last Updated: 05-05-2025
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


def gemma3_4b():
    """Load and return the Ollama model."""
    return ChatOllama(model="gemma3:4b")


def qwen3_8b(tools, base_prompt):
    """Load and return the Ollama model."""
    llm = ChatOllama(model="llama3.2", streaming=True)
    memory = InMemorySaver()

    agent = create_react_agent(
        tools=tools, model=llm, prompt=base_prompt, checkpointer=memory
    )
    return agent
