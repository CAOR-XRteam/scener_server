from langchain_core.tools import tool
from library import api

@tool
def list_assets() -> str:
    """Retrieve the list of assets in the library."""
    return api.list_asset()

@tool
def update_asset(asset_id: str, new_data: str) -> str:
    """Update an existing asset in the library."""
    return "niktamere"
