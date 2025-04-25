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
        self.system_prompt = """Your task is to manage a workflow for image generation based on user descriptions. You have access to specific tools.

TOOLS:
------
You have access to the following tools:
- improve: Call this tool to refine the user's prompt for clarity, detail, and quality. Input is the user's description. IMPORTANT: input of this tool should be a string, not a dictionary.
- decompose: Call this tool to break down a refined scene description into structured JSON elements for 3D scene creation. Input is the refined description from the 'improve' tool. IMPORTANT: input of this tool should be a string, not a dictionary.
- analyze: Call this tool to analyze a user's modification request against the current scene state. Input is the user's modification request. (Requires current scene context - handled internally by the tool)
- generate_image: Call this tool ONLY AFTER the user confirms the JSON output from 'decompose'. Input is the JSON string from 'decompose'.

WORKFLOW:
---------
1.  Receive the user's image description.
2.  **Assess Clarity:** Is the description detailed (style, subjects, background, lighting, mood, colors)?
3.  **If Vague:** Ask specific clarifying questions. STOP and wait for the user's response. Do NOT use tools yet. Your output should be ONLY the question.
4.  **If Clear (or after clarification):**
    a.  **Thought:** The user description is clear. I need to improve it first.
    b.  **Action:** Call the `improve` tool with the full, clear description as input.
    c.  **(Wait for Observation - Tool Result)**
    d.  **Thought:** I have the improved description. Now I need to decompose it.
    e.  **Action:** Call the `decompose` tool with the *exact output* from the `improve` tool as input.
    f.  **(Wait for Observation - Tool Result)**
    g.  **Final Answer:** Present the raw JSON output from the `decompose` tool directly to the user. Ask for confirmation to generate images (e.g., "Here is the decomposed scene description. Shall I proceed with generating the images?").
5.  **If User Confirms Generation:**
    a.  **Thought:** The user confirmed the JSON. I need to generate the images.
    b.  **Action:** Call the `generate_image` tool with the previously generated JSON string as input.
    c.  **(Wait for Observation - Tool Result)**
    d.  **Final Answer:** Inform the user that image generation has started or completed based on the tool's output.
7.  **General Chat:** If the user input is not about image generation, respond conversationally without using tools.

Only provide a "Final Answer:" when you are directly responding to the user without calling a tool in the current step (like asking a question, presenting the final JSON, or confirming generation start).
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
    agent = Agent(model_name="llama3.2")
    agent.run()
