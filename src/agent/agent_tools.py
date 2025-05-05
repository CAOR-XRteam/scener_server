from tools.improver import Improver
from tools.scene import SceneAnalyzer
from tools.decomposer import Decomposer
from model.black_forest import generate_image
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from beartype import beartype


class ImproveToolInput(BaseModel):
    user_input: str = Field(
        description="The user's original text prompt to be improved for clarity and detail."
    )


class DecomposeToolInput(BaseModel):
    user_input: str = Field(
        description="The improved user's scene description prompt to be decomposed."
    )


class AnalyzeToolInput(BaseModel):
    user_input: str = Field(
        description="The JSON representing extracted relevant context from the current scene state."
    )


class GenerateImageToolInput(BaseModel):
    decomposed_user_input: dict = Field(
        description="The JSON representing the decomposed scene, confirmed by the user."
    )


@beartype
class AgentTools:
    def __init__(self):
        self.improver = Improver()
        self.improve = Tool(
            name="improve",
            description="Refines the user's prompt for clarity, detail, and quality enhance the overall context.",
            args_schema=ImproveToolInput,
            func=self._improve,
        )
        self.decomposer = Decomposer()
        self.decompose = Tool(
            name="decompose",
            description="Decomposes a user's scene description prompt into manageable elements for 3D scene creation.",
            args_schema=DecomposeToolInput,
            func=self._decompose,
        )
        self.generate_image = Tool(
            name="generate_image",
            description="Generates an image based on the decomposed user's prompt using the Black Forest model.",
            args_schema=GenerateImageToolInput,
            func=self._generate_image,
        )
        self.scene_analyzer = SceneAnalyzer()
        self.analyze = Tool(
            name="analyze",
            description="Analyzes a user's modification request against the current scene state to extract relevant context or identify issues.",
            args_schema=AnalyzeToolInput,
            func=self._analyze,
        )

        self.current_scene = {}

    def get_current_scene():
        pass

    # @tool(args_schema=ImproveToolInput)
    def _improve(self, user_input: str) -> str:
        """Refines the user's prompt for clarity, detail, and quality enhance the overall context."""
        return self.improver.improve(user_input)

    # @tool(args_schema=DecomposeToolInput)
    def _decompose(self, user_input: str) -> dict:
        """Decomposes a user's scene description prompt into manageable elements for 3D scene creation."""
        return self.decomposer.decompose(user_input)

    # @tool(args_schema=AnalyzeToolInput)
    def _analyze(self, user_input: str):
        """Analyzes a user's modification request against the current scene state to extract relevant context or identify issues."""
        return self.scene_analyzer.analyze(self.get_current_scene, user_input)

    # @tool(args_schema=GenerateImageToolInput)
    def _generate_image(self, decomposed_user_input: dict):
        logging.info(f"Agent: Received decomposed user input: {decomposed_user_input}")
        """Generates an image based on the decomposed user's prompt using the Black Forest model."""
        logger.info(
            f"\nAgent: Decomposed JSON received: {decomposed_user_input}. Generating image..."
        )
        try:
            objects_to_generate = decomposed_user_input.get("scene", {}).get(
                "objects", []
            )
            logger.info(f"Agent: Decomposed objects to generate: {objects_to_generate}")
        except Exception as e:
            logger.error(f"Failed to extract objects from JSON: {e}")
            return f"[Error during image generation: {e}]"

        if not objects_to_generate:
            logger.info(
                "Agent: The decomposition resulted in no specific objects to generate images for."
            )
            return "[No objects to generate images for.]"

        for i, obj in enumerate(objects_to_generate):
            if isinstance(obj, dict) and obj.get("prompt"):
                logger.info(
                    f"Agent: Generating image for object {i+1}: {obj['prompt']}"
                )
                obj_name = obj.get("name", f"object_{i+1}").replace(" ", "_").lower()
                filename = obj_name + ".png"
                generate_image(obj["prompt"], filename)
            else:
                logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
                logger.info(f"\n[Skipping object {i+1} - missing prompt]")

        logger.info("\nAgent: Image generation process complete.")
        return f"Image generation process complete."

    def get_tools(self):
        """Returns the list of tools"""
        return [
            self.improve,
            self.decompose,
            # self.analyze,
            self.generate_image,
        ]
