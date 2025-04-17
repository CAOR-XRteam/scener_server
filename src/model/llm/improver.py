# from langchain.tools import tool
# from langchain_ollama.llms import OllamaLLM

import logging
from beartype import beartype

from ...lib import chat_call, deserialize_from_response_content

logger = logging.getLogger(__name__)


class Improver:
    @beartype
    def __init__(self, model_name):
        self.system_prompt = (
            "You are a highly skilled assistant designed to enhance and improve any given prompt. "
            "Your task is to significantly refine the prompt, providing clearer, more actionable, and "
            "detailed responses that enhance the overall context and quality. "
            "Consider the prompt from multiple angles and make sure to elevate its clarity, precision, "
            "and completeness. Avoid unnecessary details, and focus on enhancing its quality for the specific goal."
            "Example: input prompt: Generate a Japanese theatre scene with samurai armor in the center, enhanced prompt: Generate a traditional Japanese theatre scene with Samurai armor placed in the center of the stage. The room should have wooden flooring, simple red and gold accents, and folding screens in the background. The Samurai armor should be detailed, with elements like the kabuto (helmet) and yoroi (body armor), capturing the essence of a classical Japanese theatre setting."
        )
        self.model_name = model_name
        logger.info(f"Initialized with model: {self.model_name}")

    @beartype
    def improve(self, user_input: str) -> str:
        prompt = f"{self.system_prompt}\nUser: {user_input}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        return chat_call(self.model_name, messages, logger)


if __name__ == "__main__":
    improver = Improver()
    user_input = (
        "Generate a traditional Japanese theatre room with intricate wooden flooring, "
        "high wooden ceiling beams, elegant red and gold accents, and large silk curtains."
    )
    print(result=improver.improve(user_input))


# @tool
# def improver_tool(user_input: str) -> str:
#     """Refines and enhances prompts for better clarity and quality."""
#     system_prompt = (
#         "You are a highly skilled assistant designed to enhance and improve any given prompt. "
#         "Your task is to significantly refine the prompt, providing clearer, more actionable, and "
#         "detailed responses that enhance the overall context and quality. "
#         "Consider the prompt from multiple angles and make sure to elevate its clarity, precision, "
#         "and completeness. Avoid unnecessary details, and focus on enhancing its quality for the specific goal."
#     )

#     model = OllamaLLM(model="gemma3:4b", streaming=True)
#     prompt = f"{system_prompt}\nUser: {user_input}\nImproved Prompt:"
#     response_stream = model.stream(prompt)

#     result = ""
#     for chunk in response_stream:
#         result += chunk
#         print(chunk, end="", flush=True)
#     print("\n")

#     return result
