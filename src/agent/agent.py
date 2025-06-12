from agent.llm.creation import initialize_agent
from agent.tools import *
from langchain_core.tools import Tool
from lib import load_config


class Agent:
    def __init__(self):
        # Define the template for the prompt
        self.preprompt = """
You are a strict AI Workflow Manager. Your only job is to call a sequence of tools in a specific order. You do not write code or answer questions yourself. You ONLY call tools and pass data between them.

---
AVAILABLE TOOLS:
- `initial_decomposer`: Decomposes user prompt into objects.
- `improver`: Enhances prompts for each object.
- `generate_image`: Creates 2D images for each object.
- `final_decomposer`: Creates the final 3D scene JSON for Unity.
---

YOUR WORKFLOW:
You MUST follow these steps in order. Do not skip steps. Do not repeat steps.

**Step 1: Receive User Input**

**Step 2: Assess Intent**
    - If a NEW SCENE DESCRIPTION → Step 3.
    - If a MODIFICATION REQUEST → Use the `analyze` tool. (Details TBD.)
    - If GENERAL CHAT/UNRELATED → Respond using "Final Answer:".
        **Final Answer: YOUR_CONCISE_RESPONSE_STRING_HERE**
        Replace YOUR_CONCISE_RESPONSE_STRING_HERE with your direct answer. STOP.

**Step 3: Initial Decompose**
- The user will provide a scene description.
- **Thought:** I have the user's input. I must call the `initial_decomposer` tool with the user's exact `prompt`.
- **Action:** Call `initial_decomposer`. Wait for tool output.
- Store the output in a variable called `initial_decomposition`.

**Step 4: Improve Prompts**
- **Thought:** I have the `initial_decomposition`. I must now call the `improver` tool with it.
- **Action:** Call `improver` tool using the `initial_decomposition` from Step 3.
- Store the output in a variable called `improved_decomposition`. This is a critical result.

**Step 5: Generate 2D Images**
- **Thought:** I have the `improved_decomposition` from Step 4. I must now call `generate_image` tool to create the textures.
- **Action:** Call `generate_image` tool using the `improved_decomposition` from Step 4 as the `improved_decomposition` argument.
- You must still store the output from 'improver' tool you used in the step 4 in a variable called `improved_decomposition`. This is a critical result.
- Proceed to the next step without any output to the user.

**Step 6: Final Decomposition**
- **Thought:** I must now call the `final_decomposer` tool. I need two pieces of information. First, I need the `improved_decomposition` JSON that was the output of the `improver` tool in Step 4. I will look back in the conversation history to find this exact JSON output. Second, I need the very first raw message the user sent. I will look back to the beginning of the conversation to find this string. I will now combine them into a single tool call.
- **Action:** Call `final_decomposer` with the `improved_decomposition` I just found from Step 4 and the `original_user_prompt` I just found from the beginning of the conversation.
- Proceed to the next step without any output to the user.

**Step 7: Final Answer**
- **Thought:** I have successfully run all steps I must infor the user that scene decomposition and image generation are finished.
- **Action:** Output the final answer. The response MUST start with "Final Answer:" followed by your message. STOP.

If its your final answer to the user, use this format for your response:
`Final Answer: YOUR_FINAL_ANSWER_HERE`
"""
        config = load_config()

        initial_decomposer_model_name = config.get("initial_decomposer_model")
        initial_decomposer_instance = InitialDecomposer(
            model_name=initial_decomposer_model_name
        )
        initial_decomposer = Tool.from_function(
            func=initial_decomposer_instance.decompose,
            name="initial_decomposer",
            description="Converts the raw user's scene description string into a basic structured JSON, identifying key objects.",
            args_schema=InitialDecomposeToolInput,
        )

        improver_model_name = config.get("improver_model")
        improver_instance = Improver(model_name=improver_model_name)
        improver = Tool.from_function(
            func=improver_instance.improve,
            name="improver",
            description="Improves a decomposed scene description, add details and information to every component's prompt.",
            args_schema=ImproveToolInput,
        )

        final_decomposer_model_name = config.get("final_decomposer_model")
        final_decomposer_instance = FinalDecomposer(
            model_name=final_decomposer_model_name
        )
        final_decomposer = Tool.from_function(
            func=final_decomposer_instance.decompose,
            name="final_decomposer",
            description="Takes an improved scene decomposition enriches it into a full 3D scene JSON with transforms, lighting, and skybox for Unity.",
            # args_schema=FinalDecomposeToolInput,
        )

        self.tools = [
            initial_decomposer,  # OK
            improver,  # OK
            final_decomposer,
            date,  # OK
            generate_image,  # OK
            # generate_3d_object
            image_analysis,  # OK
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
