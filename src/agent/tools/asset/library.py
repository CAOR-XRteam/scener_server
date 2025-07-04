from agent.llm.creation import initialize_model
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from library.api import LibraryAPI
from loguru import logger
from colorama import Fore

from library.manager.library import Asset
from beartype import beartype


def list_assets():
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


@beartype
def find_asset_by_description(description: str) -> Asset | None:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    import json

    try:
        asset_list = list_assets()
        if not asset_list:
            return None

        assets = json.dumps([asset.model_dump() for asset in asset_list], indent=2)

        parser = JsonOutputParser(pydantic_object=Asset)

        system_prompt = """
        You are a highly precise and logical asset-matching engine. Your task is to find the single best matching 3D asset from a list, based on a target description.

        You are given a list of assets, where each asset contains the following fields: 'id', 'name', 'image', 'mesh', and 'description'. Each asset represents a 3D object and may have attributes such as size, color, material, style, and other distinctive features described in the description field.

        You are also given a separate text description of a target object.

        You must follow these rules strictly:

        1.  **Identify the Core Subject:** First, identify the primary object in the 'Target Description' (e.g., 'couch', 'cat', 'car', 'sword').

        2.  **Filter by Core Subject:** Compare this primary object with the primary object of each asset in the 'Available Assets' list. An asset is ONLY a potential match if its primary object is the SAME as the target's. A 'black cat' is NOT a match for a 'black couch'.

        3.  **Evaluate Secondary Attributes:** From the filtered list of potential matches (those with the same primary object), now evaluate secondary attributes like color, material, style, and size to find the single best fit.

        4.  **Return the Result:**
            - If you find a single asset that is a strong match on both the core subject and its attributes, return its full JSON object.
            - **If NO asset has the same core subject as the target description, you MUST return null.**
            - If there are potential matches but their secondary attributes are a poor fit, it is better to return null than to return a weak match.

        Your response must be ONLY the JSON object of the best matching asset, or the literal `null` if no sufficient match is found. Do not provide explanations or any other text."""

        user_prompt = """
        Target Description:
        {description}

        Available Assets:
        {assets}

        Instructions:
        - Compare the target description with the descriptions of all assets.
        - Return the single best matching asset.
        - If no asset matches closely enough, return null.
        - Be precise and conservative. Do not guess.

        You must respond ONLY with the JSON object of the best matching asset, or null if no match is found. Do not include any other text, explanations, or code.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        model = initialize_model("gemma3:12b")

        prompt_with_instructions = prompt.partial(
            format_instructions=parser.get_format_instructions()
        )

        chain = prompt_with_instructions | model | parser

        logger.info(f"Searching for similar asset: {description}")
        asset = chain.invoke({"description": description, "assets": assets})
        logger.info(f"Asset (not)found: {asset}")
        return Asset.model_validate(asset) if asset else None
    except Exception as e:
        logger.error(f"Error while searching for an asset: {e}")
        return None
