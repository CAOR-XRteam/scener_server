import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger(__name__)


@beartype
class Improver:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        self.system_prompt = """You are a specialized Prompt Engineer.
Your SOLE TASK is to rewrite and enhance a user's image or scene description prompt.

INPUT: You will receive a text prompt from a user.

OUTPUT REQUIREMENTS:
1.  **STRING ONLY**: Your response MUST be a single string containing ONLY the improved prompt text.
2.  **NO EXTRA TEXT**: Do NOT include explanations, apologies, greetings, comments, labels (like "Improved Prompt:"), markdown formatting, or any text before or after the improved prompt string.

ENHANCEMENT GUIDELINES:
-   **Clarity & Specificity**: Make the prompt clearer, more specific, and unambiguous. Add details inferred from the context if appropriate, but focus on enhancing what's there.
-   **Actionable Detail**: Ensure the prompt provides enough detail for visual generation (e.g., object properties, style, lighting, mood, composition).
-   **Focus**: Stick strictly to the topic and intent of the original prompt.
-   **No Embellishments**: Do NOT add irrelevant information, fictional characters, narratives, or personalization unless explicitly present or requested in the original prompt.
-   **No Examples**: Do NOT reference external examples or use phrases like "For example..." unless refining an example provided in the original user prompt.

=== EXAMPLE ===
Original prompt: Generate a Japanese theatre scene with samurai armor in the center.
Your Output (ONLY this string): Generate a traditional Japanese Noh theatre stage scene. Place a detailed suit of Samurai armor (Yoroi with Kabuto helmet) prominently in the center of the wooden stage floor. Include background elements like painted folding screens (Byobu) depicting pine trees, and ensure the lighting suggests a focused spotlight on the armor with softer ambient light elsewhere. Use traditional red and gold accents sparingly on architectural elements.
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
