import json
import logging

from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM
from model.black_forest import generate
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from beartype import beartype

logger = logging.getLogger(__name__)


class AnalyzeToolInput(BaseModel):
    user_input: str = Field(
        description="The JSON representing extracted relevant context from the current scene state."
    )


# @tool(args_schema=AnalyzeToolInput)
def _analyze(self, user_input: str):
    """Analyzes a user's modification request against the current scene state to extract relevant context or identify issues."""
    return self.scene_analyzer.analyze(self.get_current_scene, user_input)


@beartype
class SceneAnalyzer:
    def __init__(self):
        self.system_prompt = """You are an assistant that analyzes a user's request to modify a 3D scene based on its current state. Your goal is to extract **only the relevant subset** of the scene context required to understand and potentially fulfill the request.

**Task:**
Analyze the User Request provided below in the context of the Current Scene State (also provided below). Identify which objects, if any, from the Current Scene State are directly relevant to the User Request. Relevance means:
    *   The object is explicitly named or clearly described in the User Request (e.g., "the table", "the red sphere").
    *   The object is required as a reference for a relative action (e.g., "the table" in "put a lamp *on the table*").

**Output Rules:**
*   You MUST respond with **only** a single, valid JSON object.
*   The JSON object MUST follow the exact same structure as the input scene: `{{"objects": [...]}}`.
*   The `objects` list in your output JSON should contain **only** the objects identified as relevant from the input scene state. Include all original properties for those relevant objects.
*   **Crucially:**
    *   If the User Request refers to an object that does **not** exist in the Current Scene State, return `{{"objects": []}}`.
    *   If the User Request is ambiguous (e.g., refers to "the sphere" when multiple spheres exist), return `{{"objects": []}}`.
    *   If the Current Scene State is empty (`{{"objects": []}}`) and the User Request attempts to modify/reference an object, return `{{"objects": []}}`.
    *   If the User Request is purely an addition with no reference to existing objects (e.g., "add a cube"), return `{{"objects": []}}`.
*   **ABSOLUTELY NO** introductory text, explanations, apologies, notes, or markdown formatting (like ```json ```) should precede or follow the JSON output.

**Example Context for understanding the task (Do NOT replicate this structure exactly in your output, only follow the `{{"objects": [...]}}` rule):**
Input Scene Example: `{{"objects": [{{"name": "table", "position": [0,0,0]}}, {{"name": "chair", "position": [1,0,1]}}]}}`
User Prompt Example: `"Add a lamp on the table"`
Correct JSON Output based on rules: `{{"objects": [{{"name": "table", "position": [0,0,0]}}]}}`"""
        self.user_prompt = "Current scene: {current_scene}\nUser: {user_input}"
        self.model = OllamaLLM(model="llama3.1", temperature=0.0)
        self.parser = JsonOutputParser(pydantic_object=None)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )
        self.chain = self.prompt | self.model | self.parser
        # logger.info(f"Initialized with model: {model_name}")

    def analyze(self, current_scene: dict, user_input: str) -> dict:
        try:
            result: dict = self.chain.invoke(
                {"user_input": user_input, "current_scene": json.dumps(current_scene)}
            )
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


if __name__ == "__main__":
    analyzer = SceneAnalyzer()
    print(
        analyzer.analyze(
            {"objects": [{"name": "table", "position": [0, 0, 0]}]},
            "Add a lamp on the table",
        )
    )
