import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Improver:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        self.system_prompt = (
            "You are a highly skilled assistant designed to enhance and improve any given prompt. "
            "Your task is to significantly refine the prompt, providing clearer, more actionable, and "
            "detailed responses that enhance the overall context and quality. "
            "Consider the prompt from multiple angles and make sure to elevate its clarity, precision, "
            "and completeness. Avoid unnecessary details, and focus on enhancing its quality for the specific goal."
            "Example: input prompt: Generate a Japanese theatre scene with samurai armor in the center, enhanced prompt: Generate a traditional Japanese theatre scene with Samurai armor placed in the center of the stage. The room should have wooden flooring, simple red and gold accents, and folding screens in the background. The Samurai armor should be detailed, with elements like the kabuto (helmet) and yoroi (body armor), capturing the essence of a classical Japanese theatre setting."
        )

        self.user_prompt = "User: {user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = OllamaLLM(model=model_name, temperature=temperature)
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.model | self.parser

        logger.info(f"Initialized with model: {model_name}")

    def improve(self, user_input: str) -> str:
        try:
            result: str = self.chain.invoke({"user_input": user_input})
            return result
        except Exception as e:
            logger.error(f"Improvement failed: {str(e)}")
            raise


if __name__ == "__main__":
    improver = Improver()
    user_input = (
        "Generate a traditional Japanese theatre room with intricate wooden flooring, "
        "high wooden ceiling beams, elegant red and gold accents, and large silk curtains."
    )
    print(result=improver.improve(user_input))
