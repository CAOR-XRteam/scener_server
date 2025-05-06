from agent.agent import Agent
from loguru import logger
import sys

# Loguru config
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


class AgentAPI:
    def __init__(self):
        self.agent = Agent()

    def chat(self, user_input: str, thread_id: int = 0) -> str:
        self.agent.chat(user_input, thread_id)

    def run(self):
        self.agent.run()
