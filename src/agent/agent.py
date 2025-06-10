from agent.llm.creation import initialize_agent
from agent.tools import *
from langchain_core.tools import Tool
from lib import load_config


class Agent:
    def __init__(self):
        # Define the template for the prompt
        self.preprompt = """
You are a highly structured AI Workflow Manager. Your primary goal is to determine the user's intent and then execute a strict, state-based tool-calling sequence for scene generation.

---
### Phase 1: Intent Assessment

First, analyze the user's input to determine the intent.

1.  **Is it a NEW SCENE description?** (e.g., "a cat on a table", "a futuristic city")
    -   If YES, you MUST begin the Scene Generation Workflow at the START state.

2.  **Is it a MODIFICATION request?** (e.g., "make the cat orange", "add a tree")
    -   If YES, call the `analyze` tool. (This is not yet implemented, but you must know to use it).

3.  **Is it GENERAL CHAT or an unrelated question?** (e.g., "hello", "what is your name?")
    -   If YES, you must respond directly with a JSON object. Do not call any other tools.
    -   **Format:** `Final Answer: {"action": "agent_response", "message": "YOUR_CONCISE_RESPONSE"}`
    -   STOP immediately after providing this answer.

---
### Phase 2: Scene Generation Workflow (State Machine)

If the intent is a NEW SCENE, you must follow these steps sequentially. Do not repeat a step. Do not skip a step.

**Your current task is to look at the last action and determine the NEW STATE to decide what to do next.**

*   **START State**: You have only the user's input.
    -   **Action**: Call `initial_decomposer`.
    -   **Result**: A `initial_decomposition` object.
    -   **New State**: `PROMPTS_TO_IMPROVE`

*   **PROMPTS_TO_IMPROVE State**: You have the `initial_decomposition` from the previous step.
    -   **Action**: Call `improver` with the `initial_decomposition`.
    -   **Result**: An `improved_decomposition` object.
    -   **New State**: `SCENE_TO_FINALIZE`

*   **SCENE_TO_FINALIZE State**: You have the `improved_decomposition`.
    -   **Action**: Call `final_decomposer` with the `improved_decomposition`.
    -   **Result**: A `final_scene_data` object.
    -   **New State**: `IMAGES_TO_GENERATE`

*   **IMAGES_TO_GENERATE State**: You still have the `improved_decomposition`.
    -   **Action**: Call `generate_image` with the `improved_decomposition`.
    -   **Result**: An `image_generation_status` object.
    -   **New State**: `DONE`

*   **DONE State**: You have both `final_scene_data` and `image_generation_status`.
    -   **Action**: You MUST output the final, combined answer. Your response MUST start with "Final Answer:" followed by a single JSON object. Do not say anything else.
    -   **Final Answer Format**:
        `Final Answer: {"image_generation_status": <content_of_image_generation_status>, "final_scene_data": <content_of_final_scene_data>}`

---
### Quick Reference for Scene Generation:

-   If `initial_decomposer` just ran -> you are in `PROMPTS_TO_IMPROVE` state -> call `improver`.
-   If `improver` just ran -> you are in `SCENE_TO_FINALIZE` state -> call `final_decomposer`.
-   If `final_decomposer` just ran -> you are in `IMAGES_TO_GENERATE` state -> call `generate_image`.
-   If `generate_image` just ran -> you are in `DONE` state -> construct the `Final Answer`.
"""
        config = load_config()

        initial_decomposer_model_name = config.get("initial_decomposer_model")
        initial_decomposer_instance = InitialDecomposer(
            model_name=initial_decomposer_model_name
        )
        initial_decomposer_tool = Tool.from_function(
            func=initial_decomposer_instance.decompose,
            name="initial_decomposer",
            description="Converts the raw user's scene description string into a basic structured JSON, identifying key objects.",
            args_schema=InitialDecomposeToolInput,
        )

        improver_model_name = config.get("improver_model")
        improver_instance = Improver(model_name=improver_model_name)
        improver_tool = Tool.from_function(
            func=improver_instance.improve,
            name="improver",
            description="Improves a decomposed scene description, add details and information to every component's prompt.",
            args_schema=ImproveToolInput,
        )

        final_decomposer_model_name = config.get("final_decomposer_model")
        final_decomposer_instance = FinalDecomposer(
            model_name=final_decomposer_model_name
        )
        final_decomposer_tool = Tool.from_function(
            func=final_decomposer_instance.decompose,
            name="final_decomposer",
            description="Takes an initial scene decomposition with improved object prompts and enriches it into a full 3D scene JSON with transforms, lighting, and skybox for Unity.",
            args_schema=ImproveToolInput,
        )

        self.tools = [
            initial_decomposer_tool,  # OK
            improver_tool,  # OK
            final_decomposer_tool,
            date,  # OK
            generate_image,  # OK
            # generate_3d_object
            image_analysis,  # OK
            speech_to_texte,
            list_assets,
        ]

        agent_model_name = config.get("agent_model")
        self.executor = initialize_agent(agent_model_name, self.tools, self.preprompt)

    def run(self):
        from agent.llm import interaction

        interaction.run(self)

    def ask(self, query: str) -> str:
        from agent.llm import interaction

        return interaction.ask(self, query)


# Usage
if __name__ == "__main__":
    agent = Agent()
    agent.run()
