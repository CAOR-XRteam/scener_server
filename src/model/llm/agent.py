import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama.llms import OllamaLLM
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from model.llm.improver import Improver
from model.llm.scene import SceneAnalyzer
from model.llm.decomposer import Decomposer


logger = logging.getLogger(__name__)


class AgentTools:
    def __init__(self, model_name):
        self.imporver = Improver(model_name)
        self.decomposer = Decomposer(model_name)
        self.scene_analyzer = SceneAnalyzer(model_name)

    def get_current_scene():
        pass

    @tool
    def improve(self, user_input):
        """Refines the user's prompt for clarity, detail, and quality enhance the overall context."""
        return self.imporver.improve(user_input)

    @tool
    def decompose(self, user_input):
        """Decomposes a user's scene description prompt into manageable elements for 3D scene creation."""
        return self.decomposer.decompose(user_input)

    @tool
    def analyze(self, user_input):
        """Analyzes a user's modification request against the current scene state to extract relevant context or identify issues."""
        return self.scene_analyzer.analyze(self.get_current_scene, user_input)

    def get_tools(self):
        """Returns the list of tools"""
        return [self.imporver, self.decomposer, self.scene_analyzer]


class Agent:
    def __init__(self):
        # Define the template for the prompt TODO: create better prompt
        self.system_prompt = (
            "You are a helpful 3D engine chatbot assistant. "
            "Answer briefly in one or two paragraphs and do not go outside the question scope."
        )

        # Memory and Model
        self.memory = MemorySaver()
        self.model = OllamaLLM(model="llama3.2", streaming=True)
        self.tools = AgentTools(model="llama3.2")
        self.agent_executor = create_react_agent(
            self.model, self.tools, self.system_prompt, checkpointer=self.memory
        )

    def chat(self, user_input, thread_id):
        """Send a prompt to the LLM and receive a structured response."""
        agent_input = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"thread_id": thread_id}}
        final_response_content = ""

        try:
            for token in self.agent_executor.astream(
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

                print(f"\nAgent: {self.chat(user_input, thread_id=current_thread_id)}")
                current_thread_id += 1
            except KeyboardInterrupt:
                print("\nSession interrupted")
                break


# Usage
if __name__ == "__main__":
    chat = Agent()
    chat.run()
