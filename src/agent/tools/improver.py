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
        description="A text prompt to be improved for clarity and detail."
    )


class ImproveToolInput1(BaseModel):
    decomposed_input: dict = Field(
        description="A decomposed scene description ready to be improved for clarity and detail."
    )


@beartype
class Improver:
    def __init__(self, temperature: float = 0.0):
        self.system_prompt = """
            You are a specialized Prompt Engineer for 3D scene generation.

            YOUR TASK:
            - Given a user's prompt, produce a *single* improved, detailed, and clarified version of the description based uniquely on the elements of the input.

            OUTPUT FORMAT:
            - Return ONLY a single improved text string.
            - NO explanations, NO preambles, NO markdown, NO extra text.

            GUIDELINES:
            - Enhance clarity, specificity, and actionable detail (objects, material), but all details should refer to the physical aspects of element, no lighting, no weather effect.
            - You may infer reasonable details from the context if missing (e.g., typical materials).
            - NEVER invent unrelated storylines, characters, or scenes not implied by the original.
            - NEVER output anything other than the improved description string itself.
            - Provide enough details to describe a complete small scene only based on elements of the input.
            - The prompt must provide a **rich, detailed description** of the object’s physical features.
            - Focus on the object’s **key design, material, and visible features**. Avoid mentioning relationships to other objects or placement in the scene.
            - The description should be **concise but full of visual details**, ensuring that the object is clearly distinguishable and detailed for rendering.
            - The prompt must include:
            - "Placed on a white and empty background."
            - "Completely detached from surroundings."
            - **Camera view** based on object type:
            - Use "front camera view" for non-room objects.
            - Use "squared room view from the outside with a distant 3/4 top-down perspective" for room objects.
            - Example prompt:
            "A traditional Japanese theater room with detailed wooden architecture, elevated tatami stage, red silk cushions, sliding shoji panels, and ornate golden carvings on the upper walls. The room is viewed from an isometric 3/4 top-down perspective, with an open cutaway style revealing the interior. The scene is well-lit with soft, global illumination. No people, no objects outside the room, placed on a white and empty background, completely detached from surroundings."
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
def improver_bis(prompt: str) -> str:
    """Improve an input prompt, add details and information"""
    logger.info(f"Using tool {Fore.GREEN}{'improver'}{Fore.RESET}")
    tool = Improver()
    output = tool.improve(prompt)
    return output


@tool(args_schema=ImproveToolInput1)
def improver(decomposed_input: dict) -> dict:
    """Improve a decomposed scene description, add details and information to every component's prompt"""
    logger.info(f"Using tool {Fore.GREEN}{'improver'}{Fore.RESET}")
    logger.info(f"Improving decomposed scene: {decomposed_input}")
    tool = Improver()

    try:
        objects_to_improve = decomposed_input.get("scene", {}).get("objects", [])
        logger.info(f"Agent: Decomposed objects to improve: {objects_to_improve}")
    except Exception as e:
        logger.error(f"Failed to extract objects from JSON: {e}")
        return f"[Error during image generation: {e}]"

    if not objects_to_improve:
        logger.info(
            "Agent: The decomposition resulted in no specific objects to improve."
        )
        return "[No objects to improve.]"

    for i, obj in enumerate(objects_to_improve):
        if isinstance(obj, dict) and obj.get("prompt"):
            logger.info(
                f"Agent: Improving the prompt for the object {i+1}: {obj['prompt']}"
            )
            output = tool.improve(obj["prompt"])
            obj["prompt"] = output
        else:
            logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
            logger.info(f"\n[Skipping object {i+1} - missing prompt]")

    logger.info(f"Decomposed scene with enhanced prompts: {decomposed_input}")

    return decomposed_input


if __name__ == "__main__":
    user_input = "a cat on a couch in a living room"
    superprompt = improver(user_input)
    print(superprompt)
