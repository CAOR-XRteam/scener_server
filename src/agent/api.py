from agent.agent import Agent
from agent.llm.chat import chat


class AgentAPI:
    def __init__(self):
        self.agent = Agent()

    def chat(self, user_input: str, thread_id: int = 0) -> str:
        chat(self.agent, user_input, thread_id)

    def run(self):
        self.agent.run()
