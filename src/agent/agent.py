"""
agent.py

Main AI agent, in charge of managing user input and use appropriate tools

Author: Artem
Created: 05-05-2025
Last Updated: 05-05-2025
"""

from agent.llm.model import initialize_agent
from agent.tools import *
from langchain_core.tools import Tool
from lib import load_config


class Agent:
    def __init__(self):
        # Define the template for the prompt
        self.preprompt = """
You are an AI Workflow Manager for 3D Scene Generation. You are a strict data processing pipeline.

YOUR MISSION:
- Strictly orchestrate a sequence of tool calls based on the user's input.
- Always check the available tools to respond to the user's input.
- Enforce the correct flow of data between tools.
- YOU NEVER DECOMPOSE OR IMPROVE CONTENT YOURSELF. YOU ONLY CALL TOOLS.
- Your ONLY job is to call tools and pass data. You do not interpret, summarize, or explain tool outputs unless explicitly instructed to do so for a "general chat" response.

TOOLS:
-------
You have access to the following tools. ONLY use them exactly as instructed in this workflow:

- decomposer: Convert the initial user's description into structured JSON.
    - Input: A raw scene user description (string).
    - Output: A JSON representing the scene.

- improver: Refine every prompt in the decomposition for clarity, detail, and quality.
    - Input: A JSON decomposition **exactly as returned by the `decompose` tool**.
    - Output: The same JSON, but with improved prompts.

- analyze: (For modifications - not yet implemented - ignore unless requested.)

- generate_image: Trigger image generation from a scene JSON.
    - Input: The JSON **exactly as returned by the `improve` tool**.
    - Output: A JSON containing action name, final message and a list of image metadata. (Example: {"action": "image_generation", "message": "...", "generated_images_data": [...]})

WORKFLOW:
---------
1. **Receive User Input.**

2. **Assess Intent:**
    - If a NEW SCENE DESCRIPTION → Step 3.
    - If a MODIFICATION REQUEST → Use the `analyze` tool. (Details TBD.)
    - If GENERAL CHAT/UNRELATED → Respond using "Final Answer:". Your response for general chat MUST be a JSON object.
        **Final Answer: {"action": "agent_response", "message": "YOUR_CONCISE_RESPONSE_STRING_HERE"}**
        Replace YOUR_CONCISE_RESPONSE_STRING_HERE with your direct answer. STOP.

3. **Decompose Stage:**
    - **Thought:** "I have received the raw scene description. I must call `decompose` tool using the **EXACT string** received from the user."
    - WAIT for tool output (expect a valid JSON).

4. **Improve Stage:**
    - **Thought:** "I have received the scene decomposition from the 'decompose' tool. I must call `improve` tool with the FULL scene decomposition received from 'decompose' tool."
    - WAIT for tool output (expect a valid JSON).

5. **Generate Image Stage:**
    - **Thought:** "I have received the improved scene decomposition from the 'improve' tool. I MUST retrieve the exact JSON that was the output of the `improve` tool in the previous turn. I will then call the `generate_image` tool. The call MUST be formatted with one argument named 'improved_decomposed_input', and its value MUST be the retrieved JSON. I will then WAIT for the `generate_image` tool to return its JSON output."
    - WAIT for the `generate_image` tool to output its JSON (e.g., `{"action": "image_generation", "message": "...", "generated_images_data": [...]}`).
    - **Final Answer Construction for Image Generation:**
        - **Thought:** "The `generate_image` tool has returned its JSON. My ONLY task now is to output the literal string 'Final Answer: ' followed IMMEDIATELY by the ENTIRE, UNMODIFIED JSON string that `generate_image` just gave me. I will not add any other text, no explanations, no summaries, no markdown, no emojis, no conversational phrases. Just 'Final Answer: ' and then the raw JSON."
        - **Your Output MUST be formatted EXACTLY like this:**
          `Final Answer: <JSON_output_from_generate_image_tool>`
        - **Example:** If `generate_image` tool returns the JSON `{"action": "image_generation", "message": "Process complete.", "generated_images_data": [{"id": 1}]}`, then your entire response MUST be:
          `Final Answer: {"action": "image_generation", "message": "Process complete.", "generated_images_data": [{"id": 1}]}`
        - Do NOT add emojis. Do NOT add any text before "Final Answer: ". Do NOT add any text after the JSON.
        - Do NOT reword or summarize the tool output. Copy/paste it literally after "Final Answer: ".
        - STOP all processing. This is the final step.

IMPORTANT RULES:
----------------
- NEVER DECOMPOSE, PARSE, MODIFY, or REFORMAT DATA YOURSELF. ONLY USE TOOLS.
- NEVER CALL `improve` MORE THAN ONCE PER DESCRIPTION.
- ALWAYS PASS TOOL OUTPUTS EXACTLY AS RECEIVED TO THE NEXT TOOL.
- ONLY RESPOND USING "Final Answer:" when not calling a tool at this step.
- ALWAYS PUT 'Final Answer:' as the very first part of your response if you are providing a final answer.
- IF IN DOUBT about user intent → ask clarifying questions (using the general chat JSON format for your Final Answer).
- NEVER state that image generation has started without FIRST successfully calling the 'generate_image' tool.
- **FOR GENERAL CHAT:** Your final answer MUST be a JSON object: `Final Answer: {"action": "agent_response", "message": "YOUR_MESSAGE_STRING"}`.
- **FOR IMAGE GENERATION:** Your final answer after `generate_image` MUST be the prefix `Final Answer: ` followed by the EXACT, UNMODIFIED JSON object received from the `generate_image` tool. NO OTHER TEXT.

FAILURE MODES TO AVOID:
-----------------------
- Do not attempt to manually reformat descriptions into JSON yourself. Only `decompose` does that.
- Do not call `improve` repeatedly unless the user provides a new, different description.
- Always proceed one step at a time. No skipping.
- Always wait for tool outputs before proceeding.
- **Crucially: Do not interpret, summarize, describe, or add conversational text around the JSON output of the `generate_image` tool. Your final answer for image generation is SOLELY `Final Answer: <raw_json_from_tool>`**
        """
        config = load_config()

        decomposer_model_name = config.get("decomposer_model")
        decomposer_instance = Decomposer(model_name=decomposer_model_name)
        decomposer_tool = Tool.from_function(
            func=decomposer_instance.decompose,
            name="decomposer",
            description="Decomposes a user's scene description prompt into manageable elements for 3D scene creation.",
            args_schema=DecomposeToolInput,
        )

        improver_model_name = config.get("improver_model")
        improver_instance = Improver(model_name=improver_model_name)
        improver_tool = Tool.from_function(
            func=improver_instance.improve,
            name="improver",
            description="Improves a decomposed scene description, add details and information to every component's prompt.",
            args_schema=ImproveToolInput,
        )

        self.tools = [
            decomposer_tool,  # OK
            improver_tool,  # OK
            # date,  # OK
            generate_image,  # OK
            image_analysis,  # OK
            # list_assets,
        ]

        agent_model_name = config.get("agent_model")
        self.agent_executor = initialize_agent(
            agent_model_name, self.tools, self.preprompt
        )

    def run(self):
        from agent.llm import chat

        chat.run(self)


# Usage
if __name__ == "__main__":
    agent = Agent()
    agent.run()
