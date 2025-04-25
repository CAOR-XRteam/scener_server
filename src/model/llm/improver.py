import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Improver:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        self.system_prompt = """You are a skilled assistant tasked with improving prompts by making them clearer, more specific, and more actionable. 
    Your goal is to rewrite vague or generic prompts into detailed, focused instructions without adding irrelevant embellishments, fictional characters, or personalization unless explicitly asked. 
    Stick strictly to the topic of the original prompt, enhancing its clarity, structure, and descriptive power. 
    Do not use or reference any examples unless they are part of the user's prompt.\n\n
    === EXAMPLE ===\n
    Original prompt: Generate a Japanese theatre scene with samurai armor in the center.\n
    Improved prompt: Generate a traditional Japanese theatre scene with Samurai armor placed in the center of the stage. 
    Include wooden flooring, red and gold accents, and folding screens in the background. The Samurai armor should feature a kabuto (helmet) and yoroi (body armor).\n
    === END EXAMPLE ==="""

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
