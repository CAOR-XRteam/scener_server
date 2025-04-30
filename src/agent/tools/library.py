from langchain_core.tools import tool
from library import api

@tool
def library(e) -> str:
    """Read current state of the asset library"""
    return api.list_asset()
