# tests/functional/agent_tool_date.py
from agent.api import AgentAPI
from datetime import datetime
import pytest


@pytest.fixture
def llm_agent():
    def ask(prompt: str) -> str:
        api = AgentAPI()
        return api.ask(prompt)
    return ask

def test_agent_uses_date_tool(llm_agent):
    """Basic test to check if the agent uses the date tool correctly."""
    prompt = "What is today's date?"
    response = llm_agent(prompt)

    today = datetime.now().strftime("%Y-%m-%d")
    month_day = datetime.now().strftime("%B %d")

    assert today in response or month_day in response
