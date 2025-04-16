from langchain.tools import tool
from langchain_ollama.llms import OllamaLLM


@tool
def improver_tool(user_input: str) -> str:
    """Refines and enhances prompts for better clarity and quality."""
    system_prompt = (
        "You are a highly skilled assistant designed to enhance and improve any given prompt. "
        "Your task is to significantly refine the prompt, providing clearer, more actionable, and "
        "detailed responses that enhance the overall context and quality. "
        "Consider the prompt from multiple angles and make sure to elevate its clarity, precision, "
        "and completeness. Avoid unnecessary details, and focus on enhancing its quality for the specific goal."
    )

    model = OllamaLLM(model="gemma3:4b", streaming=True)
    prompt = f"{system_prompt}\nUser: {user_input}\nImproved Prompt:"
    response_stream = model.stream(prompt)

    result = ""
    for chunk in response_stream:
        result += chunk
        print(chunk, end="", flush=True)
    print("\n")

    return result
