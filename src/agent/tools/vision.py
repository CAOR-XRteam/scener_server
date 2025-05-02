from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pathlib import Path

@tool
def image_analysis(path: str) -> str:
    """Analyze image content using Gemma 3 via LangChain and Ollama."""

    image_path = Path(path).resolve()

    if not image_path.exists():
        return "Image not found."

    prompt = (
        "Provide a concise paragraph describing the visual content of the image directly and objectively, without using bullet points."
    )

    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": str(image_path)
            },
        ],
    }

    llm = ChatOllama(model="gemma3:4b")

    try:
        result = llm.invoke([message])
        return result.text()
    except Exception as e:
        return f"Image analysis failed: {str(e)}"


if __name__ == "__main__":
    a = image_analysis("image.png")
    print(a)
