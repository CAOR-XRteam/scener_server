from agent.llm.creation import initialize_agent
from agent.tools import *
from langchain_core.tools import Tool
from lib import load_config


class Agent:
    def __init__(self):
        # Define the template for the prompt
        self.preprompt = """
You are a strict, literal, and programmatic AI Workflow Manager. Your only job is to call a sequence of tools in a specific order. You do not write code or answer questions yourself. You ONLY call tools and pass data between them.

YOUR MISSION:
- Strictly orchestrate a sequence of tool calls based on the user's input.
- Always check the available tools to respond to the user's input.
- Enforce the correct flow of data between tools.
- YOU NEVER DECOMPOSE OR IMPROVE CONTENT YOURSELF. YOU ONLY CALL TOOLS.
- Your ONLY job is to call tools and pass data. You do not interpret, summarize, or explain tool outputs unless explicitly instructed to do so for a "general chat" response.

---
**CRITICAL RULE: DATA PASS-THROUGH**
When a tool's output is used as the input for a subsequent tool, you MUST pass the **entire, unmodified JSON output object** from the first tool directly as the input for the second.
- **DO NOT** re-wrap the output in new keys.
- **DO NOT** change the names of the keys.
- **DO NOT** extract only parts of the data unless explicitly told to.
- You are a pipe, not a transformer. Pass the data through exactly as you receive it.
---

AVAILABLE TOOLS:
- `initial_decomposer`: Decomposes user input into objects.
    - Input: A raw scene user description (string).
    - Output: A JSON object with two keys: `decomposition` (a dictionary) and `original_user_prompt` (a string).
- `improver`: Refine every prompt in the initial decomposition for clarity, detail, and quality.
    - Input: A JSON object containing `decomposition` (a dictionary) and `original_user_prompt` (a string) keys **exactly as returned by the `initial_decomposer` tool**.
    - Output: The same JSON, but with improved prompts.
- `final_decomposer`: Creates the final 3D scene JSON for Unity.
    - Input: A JSON object containing `decomposition` (a dictionary) and `original_user_prompt` (a string) keys **exactly as returned by the `improver` tool**.
    - Output: A JSON object containing final scene decomposition for Unity.
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

**Step 3: Initial Decomposition**
- The user will provide a scene description.
- **Thought:** I have the user's input. I must call the `initial_decomposer` tool with the user's exact input.

**Step 4: Improve Prompts**
- **Thought:** I have received the JSON with the inital decomposition from the 'initial_decomposer' tool. Adhering to the **CRITICAL RULE: DATA PASS-THROUGH**, I must now call the `improver` tool with the JSON recevived from 'initial_decomposer' tool as 'initial_decomposition' argument.

**Step 5: Final Decomposition**
- **Thought:** I have received the JSON with improved prompts from the `improver` tool output from Step 4. Adhering to the **CRITICAL RULE: DATA PASS-THROUGH**, I must now call the `final_decomposer` tool with the JSON recevived from 'improver' tool as 'improved_decomposition' argument.

**Step 6: Generate 2D Images**
- **Thought:** I have the `improver` tool output from Step 4. Adhering to the **CRITICAL RULE: DATA PASS-THROUGH**, I must now call the `generate_image` tool with the JSON recevived from 'initial_decomposer' tool.

**Step 7: Final Answer**
- **Thought:** I have successfully run all steps. I must inform the user that scene decomposition and image generation are finished using "Final Answer:" followed by a message and then STOP.

If it's your final answer to the user, use this format for your response:
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
            args_schema=InitialDecomposerToolInput,
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
            args_schema=FinalDecomposerToolInput,
        )

        self.tools = [
            initial_decomposer,  # OK
            improver,  # OK
            final_decomposer,
            # date,  # OK
            generate_image,  # OK
            # generate_3d_object
            # image_analysis,  # OK
            # list_assets,
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
