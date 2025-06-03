from agent.api import AgentAPI
from datetime import datetime
import pytest


@pytest.fixture
def llm_agent():
    def ask(prompt: str) -> dict:
        api = AgentAPI()
        return api.ask(prompt)

    return ask


def test_agent_lists_assets(llm_agent):
    prompt = "What assets are in the library?"
    ret = llm_agent(prompt)
    tools_used = [tool.lower() for tool in ret["tools"]]
    response = ret["answer"]

    # Keywords we expect to appear if the tool was called correctly
<<<<<<< HEAD:tests/functional/tool/library.py
    expected_keywords = ["Lego", ".glb", ".png", ".txt"]
=======
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

>>>>>>> c6d7108172898bc55c16ac605dd74f6b44c106dd:tests/fonctionnal/agent_tool_library.py
    for keyword in expected_keywords:
        assert keyword in response, f"Missing '{keyword}' in agent response"

    # Should call date tool only
    assert "list_assets" in tools_used
    assert len(tools_used) == 1
