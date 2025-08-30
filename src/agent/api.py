from agent.agent import Agent
from agent.llm.interaction import chat, achat
from asyncio import Queue
from beartype import beartype

# from langchain.globals import set_debug

# set_debug(True)


@beartype
class AgentAPI:
    def __init__(self):
        self.agent = Agent()

    def chat(self, user_input: str, thread_id: str = 0) -> str:
        chat(self.agent, user_input, thread_id)

    def achat(self, user_input: str, tool_output_queue: Queue, thread_id: str = 0):
        return achat(self.agent, user_input, tool_output_queue, thread_id)

    def run(self):
        self.agent.run()

    def ask(self, query: str) -> dict:
        return self.agent.ask(query)
