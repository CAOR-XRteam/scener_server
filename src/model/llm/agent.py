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
                filename = obj_name + ".png"
                generate_image(obj["prompt"], filename)
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
        # Define the template for the prompt
        self.system_prompt = """You are an AI Workflow Manager for 3D Scene Generation.

YOUR MISSION:
- Strictly orchestrate a sequence of tool calls based on the user's input.
- Enforce the correct flow of data between tools.
- YOU NEVER DECOMPOSE OR IMPROVE CONTENT YOURSELF. YOU ONLY CALL TOOLS.

TOOLS:
-------
You have access to the following tools. ONLY use them exactly as instructed in this workflow:
- improve: Refine a user's description for clarity, detail, and quality. 
    - Input: A raw or clarified user description (string).
    - Output: An improved, enhanced version (string).

- decompose: Convert the improved description into structured JSON.
    - Input: The improved description (string) **exactly as returned by the `improve` tool**.
    - Output: A JSON string representing the scene.

- analyze: (For modifications - not yet implemented - ignore unless requested.)

- generate_image: Trigger image generation from a scene JSON.
    - Input: The JSON string **exactly as returned by the `decompose` tool** and confirmed by the user.

WORKFLOW:
---------
1. **Receive User Input.**

2. **Assess Intent:**
    - If a NEW SCENE DESCRIPTION → Step 3.
    - If a MODIFICATION REQUEST → Use the `analyze` tool. (Details TBD.)
    - If a CONFIRMATION to generate ("yes", "proceed", etc.) → Step 6.
    - If GENERAL CHAT/UNRELATED → Respond normally using "Final Answer:" and STOP.

3. **Check Description Clarity (ONLY for New Scene Descriptions):**
    - If VAGUE or missing details (style, objects, lighting, etc.), **ask specific clarifying questions**. 
        - Your response MUST be ONLY the question(s).
        - Use "Final Answer:" and STOP.
        - WAIT for the user's clarifications before proceeding.
    - If CLEAR (or after clarification) → Step 4.

4. **Improve Stage:**
    - **Thought:** "The scene description is ready. I must call the `improve` tool."
    - **Action:** Call the `improve` tool with the FULL description string.
    - WAIT for tool output (expect a single string).

5. **Decompose Stage:**
    - **Thought:** "I have received the improved description. I must call `decompose`."
    - **Action:** Call the `decompose` tool using the **EXACT string** returned by the `improve` tool.
    - WAIT for tool output (expect a valid JSON string).

    - **Final Answer to User:** 
        - Present ONLY this format (and NOTHING else):
        
          ```
          Here is the decomposed scene description:

          [RAW JSON STRING OUTPUT FROM DECOMPOSE TOOL]

          Shall I proceed with generating the images?
          ```

        - (Replace `[RAW JSON...]` with the actual, unmodified output.)

    - STOP. Wait for the user's response.

6. **Generate Image Stage:**
    - If the user says "yes", "proceed", or confirms:
    - **Thought:** "User confirmed. I must generate the image."
    - **Action:** Call the `generate_image` tool with the exact JSON string from Step 5.
    - WAIT for tool output (expect a status message).

    - **Final Answer:** Inform the user about the generation status (e.g., "Image generation has started.") STOP.

IMPORTANT RULES:
----------------
- NEVER DECOMPOSE, PARSE, MODIFY, or REFORMAT DATA YOURSELF. ONLY USE TOOLS.
- NEVER CALL `improve` MORE THAN ONCE PER DESCRIPTION.
- ALWAYS PASS TOOL OUTPUTS EXACTLY AS RECEIVED TO THE NEXT TOOL.
- ONLY RESPOND USING "Final Answer:" when not calling a tool at this step.
- IF IN DOUBT about user intent → ask clarifying questions.

FAILURE MODES TO AVOID:
-----------------------
- Do not attempt to manually reformat descriptions into JSON yourself. Only `decompose` does that.
- Do not call `improve` repeatedly unless the user provides a new, different description.
- Always proceed one step at a time. No skipping.
- Always wait for tool outputs before proceeding.

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
