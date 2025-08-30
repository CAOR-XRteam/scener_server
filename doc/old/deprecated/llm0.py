from langchain_ollama import ChatOllama


def model_gemma3_4b():
    """Load and return the Ollama model."""
    return ChatOllama(model="gemma3:4b")
