from agent.agent import Agent
from beartype import beartype


@beartype
class AgentAPI:
    def __init__(self, model_name: str):
        self.agent = Agent(model_name)

    def chat(self, user_input: str, thread_id: int = 0) -> str:
        self.agent.chat(user_input, thread_id)

    def run(self):
        self.agent.run()
