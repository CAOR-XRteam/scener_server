import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Improver:
    def __init__(self, temperature: float = 0.0):
        self.system_prompt = """You are a specialized Prompt Engineer for 3D scene generation.

YOUR TASK:
- Given a user's prompt, produce a *single* improved, detailed, and clarified version of the description.

OUTPUT FORMAT:
- Return ONLY a single improved text string.
- NO explanations, NO preambles, NO markdown, NO extra text.

GUIDELINES:
- Enhance clarity, specificity, and actionable detail (objects, materials, layout, lighting, mood).
- You may infer reasonable details from the context if missing (e.g., default lighting, typical materials).
- NEVER invent unrelated storylines, characters, or scenes not implied by the original.
- NEVER output anything other than the improved description string itself."""

        self.user_prompt = "User: {user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = OllamaLLM(model="llama3.1", temperature=temperature)
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.model | self.parser

        #logger.info(f"Initialized with model: {model_name}")

    def improve(self, user_input: str) -> str:
        try:
            logger.info(f"Improving user's input: {user_input}")
            result: str = self.chain.invoke({"user_input": user_input})
            logger.info(f"Improved result: {result}")
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
    print(improver.improve(user_input))
