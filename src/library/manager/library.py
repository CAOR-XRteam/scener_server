import os
import sqlite3
import json

from beartype import beartype
from colorama import Fore
from library.sql.row import SQL
from library.manager.database import Database as DB
from loguru import logger
from pydantic import BaseModel

from agent.llm.creation import initialize_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# TODO: more precise error handling to propagate to the agent


class AppAsset(BaseModel):
    id: str
    name: str
    image: str
    mesh: str
    description: str


@beartype
class Library:
    def __init__(self, db: DB):
        self.db = db

    def fill(self, path: str):
        """Fill the database with assets from the specified directory."""
        try:
            cursor = self.db._get_cursor()  # fresh cursor
        except Exception as e:
            logger.error(f"Failed to get a connection or cursor: {e}")
            raise

        if not os.path.exists(path):
            logger.error(f"Path to fill from does not exists: {path}")
            raise FileNotFoundError(f"Path to fill from does not exists: {path}")
        if not os.path.isdir(path):
            logger.error(f"Path to fill from is not a directory: {path}")
            raise NotADirectoryError(f"Path to fill from is not a directory: {path}")

        try:
            subfolder_names = os.listdir(path)
        except OSError as e:
            logger.error(f"Failed to list directory {path}: {e}")
            raise

        for subfolder_name in subfolder_names:
            subpath = os.path.join(path, subfolder_name)
            if os.path.isdir(subpath):
                image = mesh = description = None
                try:
                    for file_name in os.listdir(subpath):
                        file_path = os.path.join(subpath, file_name)
                        absolute_file_path = os.path.abspath(file_path)

                        if file_name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".webp")
                        ):
                            image = absolute_file_path
                        elif file_name.lower().endswith(
                            (".obj", ".fbx", ".stl", ".ply", ".glb")
                        ):
                            mesh = absolute_file_path
                        elif file_name.lower().endswith(".txt"):
                            description = absolute_file_path
                    SQL.insert_asset(
                        self.db._conn, cursor, subfolder_name, image, mesh, description
                    )
                    logger.info(
                        f"Inserted asset: {Fore.GREEN}{subfolder_name}{Fore.RESET}"
                    )
                except OSError as e:
                    logger.error(f"Failed to list subdirectory {subpath}: {e}")
                except sqlite3.Error as e:
                    logger.error(f"Failed to insert asset {subfolder_name}: {e}")

    def read(self):
        """Print out all the assets in the database."""
        # Get fresh connection and cursor for querying assets
        try:
            cursor = self.db._get_cursor()
            assets = SQL.query_assets(cursor)
            if assets:
                print(
                    f"{'ID':<4} {'Name':<10} {'Image':<10} {'Mesh':<10} {'Description':<10}"
                )
                for asset in assets:
                    asset_id, asset_name, asset_image, asset_mesh, asset_description = (
                        asset
                    )
                    name = f"{Fore.YELLOW}{asset_name:<10}{Fore.RESET}"
                    img = (
                        f"{Fore.GREEN}{'ok':<10}{Fore.RESET}"
                        if asset_image
                        else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                    )
                    mesh = (
                        f"{Fore.GREEN}{'ok':<10}{Fore.RESET}"
                        if asset_mesh
                        else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                    )
                    desc = (
                        f"{Fore.GREEN}{'ok':<10}{Fore.RESET}"
                        if asset_description
                        else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                    )
                    print(f"{asset_id:<4} {name} {img} {mesh} {desc}")
            else:
                print("No assets found.")
        except Exception as e:
            logger.error(f"Failed to read assets from the database: {e}")
            raise

    def get_list(self):
        """Return a list of all assets as dictionaries."""
        # Get fresh connection and cursor for querying assets
        try:
            cursor = self.db._get_cursor()
            assets = SQL.query_assets(cursor)
            return [
                AppAsset(
                    id=str(asset_id),
                    name=name,
                    image=image,
                    mesh=mesh,
                    description=description,
                )
                for asset_id, name, image, mesh, description in assets
            ]
        except Exception as e:
            logger.error(f"Failed to read assets from the database: {e}")
            raise

    def get_asset(self, name: str):
        """Return asset by its name"""
        try:
            cursor = self.db._get_cursor()
            asset = SQL.query_asset_by_name(cursor, name)

            if asset:
                return AppAsset(
                    id=str(asset[0]),
                    name=asset[1],
                    image=asset[2],
                    mesh=asset[3],
                    description=asset[4],
                )
            else:
                raise ValueError(f"Asset {name} not found")
        except Exception as e:
            logger.error(f"Failed to get asset from the database: {e}")
            raise

    @beartype
    def find_asset_by_description(self, description: str) -> AppAsset | None:
        """Given a description, find the most corresponding asset in the database"""
        try:
            asset_list = self.get_list()
            if not asset_list:
                return None
            assets = json.dumps([asset.model_dump() for asset in asset_list])

            parser = JsonOutputParser(pydantic_object=AppAsset)

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
            prompt_with_instructions = prompt.partial(
                format_instructions=parser.get_format_instructions()
            )

            model = initialize_model("gemma3:12b")
            chain = prompt_with_instructions | model | parser

            asset = chain.invoke({"description": description, "assets": assets})

            return AppAsset.model_validate(asset) if asset else None
        except Exception as e:
            logger.error(f"Error while searching for an asset: {e}")
            return None
