"""
library_asset.py

Asset management functions

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""

from loguru import logger
from colorama import Fore
from library import sql
from library.library_database import Database as DB


class Asset:
    def __init__(self, db: DB):
        self.db = db

    def add(self, name, image=None, mesh=None, description=None):
        """Add a new asset to the database."""
        if not name:
            logger.error("Asset name is required!")
            return

        # Get a fresh connection and cursor for this operation
        conn = self.db._get_connection()
        cursor = self.db._get_cursor()

        # Check if the asset with the same name already exists
        existing_asset = self._get_asset_by_name(conn, cursor, name)
        if existing_asset:
            logger.warning(
                f"Asset with name {Fore.YELLOW}{name}{Fore.RESET} already exists."
            )
            return

        # Insert the new asset
        sql.insert_asset(conn, cursor, name, image, mesh, description)
        logger.success(f"Asset {Fore.GREEN}{name}{Fore.RESET} added successfully.")

    def delete(self, name):
        """Delete an asset by its name."""
        if not name:
            logger.error("Asset name is required for deletion!")
            return

        # Get a fresh connection and cursor for this operation
        conn = self.db._get_connection()
        cursor = self.db._get_cursor()

        # Check if the asset exists
        asset = self._get_asset_by_name(conn, cursor, name)
        if not asset:
            logger.warning(f"Asset {Fore.RED}{name}{Fore.RESET} not found.")
            return

        # Delete the asset
        sql.delete_asset(conn, cursor, name)
        logger.success(f"Asset {Fore.GREEN}{name}{Fore.RESET} deleted successfully.")

    def update(self, name, image=None, mesh=None, description=None):
        """Update an existing asset."""
        if not name:
            logger.error("Asset name is required for update!")
            return

        # Get a fresh connection and cursor for this operation
        conn = self.db._get_connection()
        cursor = self.db._get_cursor()

        # Check if the asset exists
        asset = self._get_asset_by_name(conn, cursor, name)
        if not asset:
            logger.warning(f"Asset {Fore.RED}{name}{Fore.RESET} not found.")
            return

        # Update the asset
        sql.update_asset(conn, cursor, name, image, mesh, description)
        logger.success(f"Asset {Fore.GREEN}{name}{Fore.RESET} updated successfully.")

    def _get_asset_by_name(self, conn, cursor, name):
        """Helper method to fetch an asset by its name."""
        cursor.execute("SELECT * FROM asset WHERE name = ?", (name,))
        return cursor.fetchone()
