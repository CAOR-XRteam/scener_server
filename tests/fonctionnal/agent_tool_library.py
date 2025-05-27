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


def test_agent_lists_assets(llm_agent):
    prompt = "What assets are in the library?"
    response = llm_agent(prompt)

    # Keywords we expect to appear if the tool was called correctly
    expected_keywords = [
        "Theatre",
        "Robot",
        "Lego",
        "Astronaut",
        "Samurai",
        ".glb",
        ".webp",
        ".png",
        ".txt",
    ]

    for keyword in expected_keywords:
        assert keyword in response, f"Missing '{keyword}' in agent response"
