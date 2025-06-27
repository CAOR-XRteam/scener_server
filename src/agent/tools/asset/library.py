from langchain_core.tools import tool
from library.api import LibraryAPI
from loguru import logger
from colorama import Fore


@tool
def list_assets() -> str:
    """Retrieve the library list of assets, containing their element (image, description, mesh) paths in the SQL database."""
    api = LibraryAPI()
    return api.get_list()


@tool
def update_asset(
    name: str, image_path: str, mesh_path: str, description_path: str
) -> str:
    """Update an existing asset by name with image path, mesh path, and description path."""
    api.update_asset(name, image_path, mesh_path, description_path)
    return "asset updated"


@tool
def create_description_file(path_with_name: str, asset_description: str) -> str:
    """Create a text file with the given content. The name of the file should be the same than the asset name"""
    try:
        with open(path_with_name, "w") as file:
            file.write(asset_description)
        return f"Text file '{path_with_name}' created successfully."
    except Exception as e:
        return f"Failed to create text file: {str(e)}"
