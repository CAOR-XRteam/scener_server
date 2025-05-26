from agent.llm.creation import initialize_model
from colorama import Fore
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib import logger
from pydantic import BaseModel, Field


class ImproveToolInput(BaseModel):
    decomposed_input: dict = Field(
        description="A decomposed scene description ready to be improved for clarity and detail."
    )


@beartype
class Improver:
    def __init__(self, model_name: str, temperature: float = 0.0):
        # TODO: mandatory room? if other type of background?
        self.system_prompt = """
            You are a specialized Prompt Engineer for 3D object generation.

            YOUR TASK:
            - Given a user's prompt for a single object, produce an improved, detailed, and clarified version of its description.
            - Focus *exclusively* on the physical aspects of the object itself.

            OUTPUT FORMAT:
            - Return ONLY the improved text string for the object.
            - NO explanations, NO preambles, NO markdown, NO extra text.

            GUIDELINES:
            - Enhance clarity, specificity, and actionable detail regarding the object's design, material, texture, color, and key visible features.
            - All details MUST refer to the physical aspects of the object. DO NOT include lighting, weather effects, or its relationship to other objects or placement in a larger scene (beyond the required background/camera statements).
            - You may infer reasonable details from the context if missing (e.g., typical materials for the object type).
            - NEVER invent unrelated storylines, characters, or scenes not implied by the original object prompt.
            - NEVER output anything other than the improved description string itself.
            - The prompt must provide a **rich, detailed description** of the object’s physical features.
            - The prompt must include:
                - "Placed on a white and empty background."
                - "Completely detached from surroundings."
            - **Camera view** based on object type:
                - For non-room objects (e.g., 'prop', 'furniture', 'character'): Use "front camera view".
                - For room objects (e.g., 'room', 'environment'): Use "squared room view from the outside with a distant 3/4 top-down perspective".

            EXAMPLE OF AN IMPROVED *OBJECT* PROMPT (for a non-room object):
            Original: "a red chair"
            Improved: "A vibrant red armchair, crafted from polished mahogany wood, featuring a high back with button-tufted detailing, plush velvet cushioning on the seat and backrest, and elegantly curved cabriole legs. Front camera view. Placed on a white and empty background. Completely detached from surroundings."

            EXAMPLE OF AN IMPROVED *ROOM* PROMPT:
            Original: "a kitchen"
            Improved: "A modern, minimalist kitchen with sleek white cabinetry, stainless steel appliances including a double-door refrigerator and a built-in oven, a central island with a quartz countertop and induction cooktop, and light gray porcelain tile flooring. Squared room view from the outside with a distant 3/4 top-down perspective. Placed on a white and empty background. Completely detached from surroundings."
            """
        self.user_prompt = "User: {user_input}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.model = initialize_model(model_name, temperature=temperature)
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.model | self.parser

    def improve_single_prompt(self, prompt: str) -> str:
        try:
            logger.info(f"Improving user's input: {prompt}")
            result: str = self.chain.invoke({"user_input": prompt})
            logger.info(f"Improved result: {result}")
            return result
        except Exception as e:
            logger.error(f"Improvement failed: {str(e)}")
            raise

    def improve(self, decomposed_input: dict) -> dict:
        """Improve a decomposed scene description, add details and information to every component's prompt"""
        logger.info(f"Improving decomposed scene: {decomposed_input}")

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
                output = self.improve_single_prompt(obj["prompt"])
                obj["prompt"] = output
            else:
                logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
                logger.info(f"\n[Skipping object {i+1} - missing prompt]")

        logger.info(f"Decomposed scene with enhanced prompts: {decomposed_input}")

        return decomposed_input


if __name__ == "__main__":
    user_input = "a cat on a couch in a living room"
    improver = Improver(model_name="llama3.1")
    superprompt = improver.improve_single_prompt(user_input)
    print(superprompt)
