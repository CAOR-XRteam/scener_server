from agent import Agent


class AgentAPI:
    def __init__(self):
        self.agent = Agent()

    def chat(self, user_input: str, thread_id: int = 0) -> str:
        self.agent.chat(user_input, thread_id)

    def run(self):
        self.agent.run()
