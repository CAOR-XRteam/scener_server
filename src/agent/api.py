from agent.agent import Agent
from agent.llm import chat
from beartype import beartype
from loguru import logger
import sys

# Loguru config
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@beartype
class AgentAPI:
    def __init__(self):
        self.agent = Agent()

    def chat(self, user_input: str, thread_id: str = 0) -> str:
        chat(self.agent, user_input, thread_id)

    def run(self):
        self.agent.run()
