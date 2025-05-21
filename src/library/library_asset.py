# TODO: more precise error handling to propagate to the agent


import sqlite3

from beartype import beartype
from colorama import Fore
from library.sql import Sql
from library.library_database import Database as DB
from loguru import logger


@beartype
class Asset:
    def __init__(self, db: DB):
        self.db = db

    def add(
        self, name: str, image: str = None, mesh: str = None, description: str = None
    ):
        """Add a new asset to the database."""
        if not name:
            logger.error("Asset name is required for addition!")
            raise ValueError("Asset name is required for addition!")

        # Get a fresh connection and cursor for this operation
        try:
            # Check if the asset with the same name already exists
            cursor = self.db._get_cursor()
            existing_asset = self._get_asset_by_name(cursor, name)
            if existing_asset:
                logger.error(
                    f"Asset with name {Fore.YELLOW}'{name}'{Fore.RESET} already exists."
                )
                raise ValueError(
                    f"Asset with name {Fore.YELLOW}'{name}'{Fore.RESET} already exists."
                )
            # Insert the new asset
            Sql.insert_asset(self.db._conn, cursor, name, image, mesh, description)
            logger.success(
                f"Asset {Fore.GREEN}'{name}'{Fore.RESET} added successfully."
            )
        except ValueError as ve:
            raise
        except Exception as e:
            logger.error(f"Failed to add the asset '{name}': {e}")
            raise

    def delete(self, name: str):
        """Delete an asset by its name."""
        if not name:
            logger.error("Asset name is required for deletion!")
            raise ValueError("Asset name is required for deletion!")

        # Get a fresh connection and cursor for this operation
        try:
            cursor = self.db._get_cursor()

            # Check if the asset exists
            asset = self._get_asset_by_name(cursor, name)
            if not asset:
                logger.warning(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")
                raise ValueError(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")

            # Delete the asset
            Sql.delete_asset(self.db._conn, cursor, name)
            logger.success(
                f"Asset {Fore.GREEN}'{name}'{Fore.RESET} deleted successfully."
            )
        except ValueError as ve:
            raise
        except Exception as e:
            logger.error(f"Failed to delete the asset '{name}': {e}")
            raise

    def update(
        self, name: str, image: str = None, mesh: str = None, description: str = None
    ):
        """Update an existing asset."""
        if not name:
            logger.error("Asset name is required for update!")
            raise ValueError("Asset name is required for update!")

        # Get a fresh connection and cursor for this operation
        try:
            cursor = self.db._get_cursor()

            # Check if the asset exists
            asset = self._get_asset_by_name(cursor, name)
            if not asset:
                logger.warning(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")
                raise ValueError(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")
            # Update the asset
            Sql.update_asset(self.db._conn, cursor, name, image, mesh, description)
            logger.success(
                f"Asset {Fore.GREEN}'{name}'{Fore.RESET} updated successfully."
            )
        except ValueError as ve:
            raise
        except Exception as e:
            logger.error(f"Failed to update the asset '{name}': {e}")
            raise

    def _get_asset_by_name(self, cursor: sqlite3.Cursor, name: str):
        """Helper method to fetch an asset by its name."""
        try:
            cursor.execute("SELECT * FROM asset WHERE name = ?", (name,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch asset '{name}': {e}")
            raise
