from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


browsing_tool = DuckDuckGoSearchRun()

@tool
def browsing(query: str) -> str:
    """Search for general knowledge using DuckDuckGo. Input should be a single search term."""
    return wikipedia_tool.run(query)
