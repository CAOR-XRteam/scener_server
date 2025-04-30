from langchain_community.llms import Ollama


def load_llm():
    """Load and return the Ollama model."""
    return Ollama(model="gemma3:4b")
