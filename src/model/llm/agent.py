import logging

from beartype import beartype
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from improver import Improver
from scene import SceneAnalyzer
from decomposer import Decomposer

from utils.json import convert
from model.generation.black_forest import generate_image
from lib import setup_logging


logger = logging.getLogger(__name__)


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
    decomposed_user_input: str = Field(
        description="The JSON representing the decomposed scene, confirmed by the user."
    )


@beartype
class AgentTools:
    def __init__(self, model_name: str):
        self.improver = Improver(model_name)
        self.improve = Tool(
            name="improve",
            description="Refines the user's prompt for clarity, detail, and quality enhance the overall context.",
            args_schema=ImproveToolInput,
            func=self.improver.improve,
        )
        self.decomposer = Decomposer(model_name)
        self.decompose = Tool(
            name="decompose",
            description="Decomposes a user's scene description prompt into manageable elements for 3D scene creation.",
            args_schema=DecomposeToolInput,
            func=self.decomposer.decompose,
        )
        self.generate_image = Tool(
            name="generate_image",
            description="Generates an image based on the decomposed user's prompt using the Black Forest model.",
            args_schema=GenerateImageToolInput,
            func=self._generate_image,
        )
        self.scene_analyzer = SceneAnalyzer(model_name)
        self.analyze = Tool(
            name="analyze",
            description="Analyzes a user's modification request against the current scene state to extract relevant context or identify issues.",
            args_schema=AnalyzeToolInput,
            func=self.scene_analyzer.analyze,
        )

        self.current_scene = {}

    def get_current_scene():
        pass

    # @tool(args_schema=ImproveToolInput)
    def _improve(self, user_input: str):
        """Refines the user's prompt for clarity, detail, and quality enhance the overall context."""
        return self.improver.improve(user_input)

    # @tool(args_schema=DecomposeToolInput)
    def _decompose(self, user_input: str):
        """Decomposes a user's scene description prompt into manageable elements for 3D scene creation."""
        return self.decomposer.decompose(user_input)

    # @tool(args_schema=AnalyzeToolInput)
    def _analyze(self, user_input: str):
        """Analyzes a user's modification request against the current scene state to extract relevant context or identify issues."""
        return self.scene_analyzer.analyze(self.get_current_scene, user_input)

    # @tool(args_schema=GenerateImageToolInput)
    def _generate_image(self, decomposed_user_input: dict):
        """Generates an image based on the decomposed user's prompt using the Black Forest model."""
        # try:
        #     parsed_response = convert(decomposed_user_input)
        # except Exception as e:
        #     logger.error(f"Failed to parse JSON: {e}")
        #     raise

        logger.info("\nAgent: Decomposition received. Generating images...")
        objects_to_generate = decomposed_user_input.get("scene", {}).get("objects", [])
        logger.debug(f"Agent: Decomposed objects to generate: {objects_to_generate}")

        if not objects_to_generate:
            logger.info(
                "Agent: The decomposition resulted in no specific objects to generate images for."
            )
            return

        for i, obj in enumerate(objects_to_generate):
            if isinstance(obj, dict) and obj.get("prompt"):
                obj_name = obj.get("name", f"object_{i+1}").replace(" ", "_").lower()
                safe_filename = obj_name + ".png"
                if not safe_filename:
                    safe_filename = f"object_{i+1}.png"
                generate_image(prompt=obj["prompt"], filename=safe_filename)
            else:
                logger.warning(f"Skipping object due to missing/empty prompt: {obj}")
                logger.info(f"\n[Skipping object {i+1} - missing prompt]")

        logger.info("\nAgent: Image generation process complete.")

    def get_tools(self):
        """Returns the list of tools"""
        return [
            self.improve,
            self.decompose,
            # self.analyze,
            self.generate_image,
        ]


@beartype
class Agent:
    def __init__(self, model_name: str = "llama3.2"):
        # Define the template for the prompt TODO: create better prompt
        self.system_prompt = """You are an AI Workflow Manager for 3D Scene Generation. Your primary function is to orchestrate a sequence of tool calls based on user input, ensuring data flows correctly between tools.

TOOLS:
------
You have access to the following tools. Use them *only* as specified in the workflow.
- improve: Call this tool to refine the user's prompt for clarity, detail, and quality.
    - Input: The user's raw description (string).
    - Output: An improved description (string).
- decompose: Call this tool to break down a refined scene description into structured JSON.
    - Input: The refined description string *exactly* as output by the 'improve' tool.
    - Output: A JSON string representing the scene structure.
- analyze: (Use for modification requests - details omitted for brevity based on original prompt focus)
- generate_image: Call this tool to trigger image generation.
    - Input: The JSON string *exactly* as output by the 'decompose' tool and confirmed by the user.
    - Output: Confirmation or status of image generation.

WORKFLOW & STRICT RULES:
-----------------------
1.  **Receive Input:** Get the user's request.
2.  **Assess Intent:**
    *   **Is it a NEW scene description?** Proceed to Step 3.
    *   **Is it a modification request?** Use the `analyze` tool (if applicable). (Workflow for this path TBD based on `analyze` tool needs). #TODO: modify once analyze is implemented 
    *   **Is it a confirmation to generate ('yes', 'proceed', etc.) AFTER you presented the JSON?** Proceed to Step 6.
    *   **Is it general conversation or unrelated?** Respond conversationally WITHOUT using any tools. Provide the response directly using "Final Answer:". Stop.
3.  **Assess Clarity (for NEW descriptions):** Does the description seem vague or lack key details (e.g., style, specific objects, layout, lighting, mood, colors) likely needed for good generation?
    *   **If Vague:** Ask specific clarifying questions to get necessary details. Your response MUST be ONLY the question(s). Use "Final Answer:". STOP and wait for the user's response. Do NOT use tools yet.
    *   **If Clear (or after clarification):** Proceed to Step 4.
4.  **Improve Stage:**
    *   **Thought:** The description is ready. I must call the `improve` tool to enhance it.
    *   **Action:** Call the `improve` tool. The input MUST be the user's full, clear description string.
    *   **(Wait for Observation - Tool Result: Expecting an improved string)**
5.  **Decompose Stage:**
    *   **Thought:** I have received the improved description string from the `improve` tool. I must now call the `decompose` tool using this exact string.
    *   **Action:** Call the `decompose` tool. CRITICAL: The input MUST be the *exact, unmodified string* received as output from the `improve` tool in the previous step. You MUST use the 'decompose' tool and not decompose the string yourself.
    *   **(Wait for Observation - Tool Result: Expecting a JSON string)**
    *   **Final Answer:** Present the result to the user. Your response MUST be ONLY the following format: "Here is the decomposed scene description:\n\n[RAW JSON STRING OUTPUT FROM DECOMPOSE TOOL]\n\nShall I proceed with generating the images?" (Replace "[RAW JSON STRING...]" with the actual, unmodified JSON string from the tool). Do NOT add any other commentary or formatting around the JSON. STOP.
6.  **Generate Stage (User Confirmed):**
    *   **Thought:** The user confirmed the JSON. I must call the `generate_image` tool using the previously generated JSON string.
    *   **Action:** Call the `generate_image` tool. The input MUST be the *exact JSON string* that was output by the `decompose` tool and presented to the user.
    *   **(Wait for Observation - Tool Result: Expecting generation status)**
    *   **Final Answer:** Inform the user about the image generation status based on the tool's output (e.g., "Image generation has started." or "Image generation complete."). STOP.

**CRITICAL REMINDERS:**
*   **Follow the Workflow Strictly:** Do not skip steps or call tools out of order.
*   **Exact Data Transfer:** Pass outputs from one tool as inputs to the next *exactly* as received, without modification, unless the workflow explicitly states otherwise.
*   **Tool Input Types:** Respect the specified input types (string vs. JSON string) for each tool.
*   **"Final Answer:" Usage:** Only use "Final Answer:" when you are directly responding to the user *without* calling a tool in the *current* step (Steps 2-General Chat, 3-Vague, 5-Present JSON, 6-Inform Status).
*   **No Assumptions:** If unsure about the user's intent or if a description is too vague, ask for clarification (Step 3). Do not guess or proceed with ambiguous input.
        """

        # Memory and Model
        self.memory = MemorySaver()
        self.tools = AgentTools(model_name)

        self.model = ChatOllama(model=model_name, streaming=True)
        self.model.invoke("Hello")
        logger.info(f"ChatOllama model '{model_name}' initialized successfully.")

        self.agent_executor = create_react_agent(
            self.model,
            self.tools.get_tools(),
            prompt=self.system_prompt,
            checkpointer=self.memory,
        )

    def chat(self, user_input: str, thread_id: int = 0):
        """Send a prompt to the LLM and receive a structured response."""
        agent_input = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"thread_id": thread_id}}
        final_response_content = ""

        try:
            for token in self.agent_executor.stream(
                agent_input, config=config, stream_mode="values"
            ):
                last_message = token["messages"][-1]

                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    new_content = last_message.content[len(final_response_content) :]
                    if new_content:
                        print(new_content, end="")
                        final_response_content += new_content
        except Exception as e:
            logger.info(f"\nAgent error occurred: {e}")
            return f"[Error during agent execution: {e}]"

        return final_response_content

    def run(self):
        print("Type 'exit' to quit")
        current_thread_id = 0

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    print("bye")
                    break
                if not user_input.strip():
                    continue

                print("Agent: ")

                self.chat(user_input, thread_id=current_thread_id)

            except KeyboardInterrupt:
                print("\nSession interrupted")
                break


# Usage
if __name__ == "__main__":
    setup_logging()
    model_name = input("Enter the model name: ").strip() or "llama3.2"
    agent = Agent(model_name)
    agent.run()
