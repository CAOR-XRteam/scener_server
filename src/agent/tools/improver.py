from loguru import logger
from colorama import Fore
from pydantic import BaseModel, Field
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import tool


class ImproveToolInput(BaseModel):
    prompt: str = Field(
        description="The user's original text prompt to be improved for clarity and detail."
    )

@beartype
class Improver:
    def __init__(self, temperature: float = 0.0):
        self.system_prompt ="""
            You are a specialized Prompt Engineer for 3D scene generation.

            YOUR TASK:
            - Given a user's prompt, produce a *single* improved, detailed, and clarified version of the description.

            OUTPUT FORMAT:
            - Return ONLY a single improved text string.
            - NO explanations, NO preambles, NO markdown, NO extra text.

            GUIDELINES:
            - Enhance clarity, specificity, and actionable detail (objects, materials, layout, lighting, mood).
            - You may infer reasonable details from the context if missing (e.g., default lighting, typical materials).
            - NEVER invent unrelated storylines, characters, or scenes not implied by the original.
            - NEVER output anything other than the improved description string itself.
            """

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

    def improve(self, user_input: str) -> str:
        try:
            logger.info(f"Improving user's input: {user_input}")
            result: str = self.chain.invoke({"user_input": user_input})
            logger.info(f"Improved result: {result}")
            return result
        except Exception as e:
            logger.error(f"Improvement failed: {str(e)}")
            raise

@tool(args_schema=ImproveToolInput)
def improver(prompt: str) -> str:
    """Improve an input prompt, add details and information"""
    logger.info(f"Using tool {Fore.GREEN}{'improver'}{Fore.RESET}")
    tool = Improver()
    output = tool.improve(prompt)
    return output

if __name__ == "__main__":
    user_input = (
        "a cat in a house"
    )
    superprompt = improver(user_input)
    print(superprompt)
