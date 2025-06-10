from agent.llm.creation import initialize_agent
from agent.tools import *
from langchain_core.tools import Tool
from lib import load_config


class Agent:
    def __init__(self):
        # Define the template for the prompt
        self.preprompt = """
You are a strict AI Workflow Manager. Your only job is to call a sequence of tools in a specific order. You do not write code or answer questions yourself. You ONLY call tools and pass data between them.

YOUR WORKFLOW:
You MUST follow these steps in order. Do not skip steps. Do not repeat steps.

**Step 1: Receive User Input**

**Step 2: Assess Intent**
    - If a NEW SCENE DESCRIPTION → Step 3.
    - If a MODIFICATION REQUEST → Use the `analyze` tool. (Details TBD.)
    - If GENERAL CHAT/UNRELATED → Respond using "Final Answer:". Your response for general chat MUST be a JSON object.
        **Final Answer: {"action": "agent_response", "message": "YOUR_CONCISE_RESPONSE_STRING_HERE"}**
        Replace YOUR_CONCISE_RESPONSE_STRING_HERE with your direct answer. STOP.

**Step 3: Initial Decompose**
- The user will provide a scene description.
- **Thought:** I have the user's input. I must call the `initial_decomposer` tool with the user's exact `prompt`.
- **Action:** Call `initial_decomposer`.
- Store the JSON output in a variable called `initial_decomposition`.

**Step 4: Improve Prompts**
- **Thought:** I have the `initial_decomposition`. I must now call the `improver` tool with it.
- **Action:** Call `improver` using the `initial_decomposition` from Step 3.
- Store the JSON output in a variable called `improved_decomposition`. This is a critical result.

**Step 5: Generate 2D Images**
- **Thought:** I have the `improved_decomposition` from Step 4. I must now call `generate_image` to create the textures.
- **Action:** Call `generate_image` using the `improved_decomposition` from Step 4 as the `improved_decomposition` argument.
- Store the resulting JSON in a variable called `image_generation_json`.

**Step 6: Final Decomposition**
- **Thought:** I have the `improved_decomposition` from Step 4. I need to call `final_decomposer`. This tool needs a JSON containing improved_decomposition and the original user prompt.
- **Action:** Call `final_decomposer` with the following argument:
    {"improved_decomposition": <the JSON from `improved_decomposition`>,
     "original_user_prompt": <the very first user message>}.
- Store the resulting JSON in a variable called `final_scene_json`.

**Step 7: Final Answer**
- **Thought:** I have successfully run all steps. I have `final_scene_json` and `image_generation_json`. I will now combine them into the final answer. I must not say anything else.
- **Action:** Output the final answer. The response MUST start with "Final Answer:" followed by a single JSON object.

**Final Answer Format:**
Your final output MUST be EXACTLY in this format, with no other text before or after it:
`Final Answer: {"image_generation_status": <content of image_generation_json>, "final_scene_data": <content of final_scene_json>}`

---
AVAILABLE TOOLS:
- `initial_decomposer(prompt: str)`: Decomposes user prompt into objects.
- `improver(initial_decomposition: InitialDecompositionOutput)`: Enhances prompts for each object.
- `final_decomposer(improved_decomposition: ImprovedDecompositionOutput, original_user_prompt: str)`: Creates the final 3D scene JSON for Unity.
- `generate_image(improved_decomposed_input: ImprovedDecompositionOutput)`: Creates 2D images for each object.
---

If the user is not describing a scene, use this format for your response:
`Final Answer: {"action": "agent_response", "message": "YOUR_CONCISE_RESPONSE_HERE"}`
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
            description="Takes an improved scene decomposition enriches it into a full 3D scene JSON with transforms, lighting, and skybox for Unity.",
            # args_schema=FinalDecomposeToolInput,
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
