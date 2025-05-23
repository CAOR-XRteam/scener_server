# TODO: more precise error handling to propagate to the agent

from beartype import beartype
from colorama import Fore
from library.sql.row import SQL
from library.manager.database import Database as DB
from loguru import logger
import sqlite3
import os


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

                        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
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
                {
                    "id": asset_id,
                    "name": name,
                    "image": image,
                    "mesh": mesh,
                    "description": description,
                }
                for asset_id, name, image, mesh, description in assets
            ]
        except Exception as e:
            logger.error(f"Failed to read assets from the database: {e}")
            raise
