"""
library_asset.py

Asset management functions

Author: Nathan SV
Created: 05-05-2025
Last Updated: 19-05-2025
"""

# TODO: more precise error handling to propagate to the agent


import sqlite3

from colorama import Fore
from library.sql import Sql
from library.library_database import Database as DB
from loguru import logger


class Asset:
    def __init__(self, db: DB):
        self.db = db

    def add(self, name, image=None, mesh=None, description=None):
        """Add a new asset to the database."""
        if not name:
            logger.error("Asset name is required!")
            raise ValueError("Asset name is required!")

        # Get a fresh connection and cursor for this operation
        try:
            # Check if the asset with the same name already exists
            cursor = self.db._get_cursor()
            existing_asset = self._get_asset_by_name(cursor, name)
            if existing_asset:
                logger.warning(
                    f"Asset with name {Fore.YELLOW}{name}{Fore.RESET} already exists."
                )
                raise ValueError(
                    f"Asset with name {Fore.YELLOW}{name}{Fore.RESET} already exists."
                )
            try:
                # Insert the new asset
                Sql.insert_asset(self.db._conn, cursor, name, image, mesh, description)
                logger.success(
                    f"Asset {Fore.GREEN}{name}{Fore.RESET} added successfully."
                )
            except Exception as e:
                logger.error(f"Failed to add asset {name}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to get a cursor: {e}")
            raise

    def delete(self, name):
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
                logger.warning(f"Asset {Fore.RED}{name}{Fore.RESET} not found.")
                raise ValueError(f"Asset {Fore.RED}{name}{Fore.RESET} not found.")

            # Delete the asset
            try:
                Sql.delete_asset(self.db._conn, cursor, name)
                logger.success(
                    f"Asset {Fore.GREEN}{name}{Fore.RESET} deleted successfully."
                )
            except Exception as e:
                logger.error(f"Failed to delete asset {name}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to get a cursor: {e}")
            raise

    def update(self, name, image=None, mesh=None, description=None):
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
                logger.warning(f"Asset {Fore.RED}{name}{Fore.RESET} not found.")
                raise ValueError(f"Asset {Fore.RED}{name}{Fore.RESET} not found.")
            # Update the asset
            try:
                Sql.update_asset(self.db._conn, cursor, name, image, mesh, description)
                logger.success(
                    f"Asset {Fore.GREEN}{name}{Fore.RESET} updated successfully."
                )
            except Exception as e:
                logger.error(f"Failed to update asset {name}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to get a cursor: {e}")
            raise

    def _get_asset_by_name(self, cursor, name):
        """Helper method to fetch an asset by its name."""
        try:
            cursor.execute("SELECT * FROM asset WHERE name = ?", (name,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch asset {name}: {e}")
            raise
