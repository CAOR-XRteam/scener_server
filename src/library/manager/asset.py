from pathlib import Path
from beartype import beartype
from colorama import Fore
from library.sql.row import SQL
from library.manager.database import Database as DB
from loguru import logger


@beartype
class Asset:
    def __init__(self, db: DB):
        self.db = db

        """ A mettre qql part de mieux """
        from library import path_asset
        from library.manager.library import Library
        import os

        if not os.path.exists(path_asset):
            os.makedirs(path_asset)
        library = Library(db)
        library.fill(path_asset)

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
            existing_asset = SQL.query_asset_by_name(cursor, name)
            if existing_asset:
                logger.error(
                    f"Asset with name {Fore.YELLOW}'{name}'{Fore.RESET} already exists."
                )
                raise ValueError(
                    f"Asset with name {Fore.YELLOW}'{name}'{Fore.RESET} already exists."
                )
            # Insert the new asset
            SQL.insert_asset(self.db._conn, cursor, name, image, mesh, description)
            logger.success(
                f"Asset {Fore.GREEN}'{name}'{Fore.RESET} added successfully."
            )
        except ValueError as ve:
            raise
        except Exception as e:
            logger.error(f"Failed to add the asset '{name}': {e}")
            raise

    def _delete_local_asset_files(self, name: str):
        """
        Deletes the .glb and .png files associated with an asset name
        """
        media_path = Path("src/media/temp")
        if not media_path.is_dir():
            logger.warning(
                f"Media directory '{media_path}' not found. Skipping file deletion."
            )
            return

        extensions_to_delete = [".glb", ".png"]

        for ext in extensions_to_delete:
            file_path = media_path / f"{name}{ext}"
            try:
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"Deleted local file: {file_path}")
            except OSError as e:
                logger.error(f"Error deleting file {file_path}: {e}")

    def delete(self, name: str):
        """Delete an asset by its name."""
        if not name:
            logger.error("Asset name is required for deletion!")
            raise ValueError("Asset name is required for deletion!")

        # Get a fresh connection and cursor for this operation
        try:
            cursor = self.db._get_cursor()

            # Check if the asset exists
            asset = SQL.query_asset_by_name(cursor, name)
            if not asset:
                logger.warning(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")
                raise ValueError(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")

            # Delete the asset
            SQL.delete_asset(self.db._conn, cursor, name)
            logger.success(
                f"Asset {Fore.GREEN}'{name}'{Fore.RESET} deleted successfully."
            )
            self._delete_local_asset_files(name)
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
            asset = SQL.query_asset_by_name(cursor, name)
            if not asset:
                logger.warning(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")
                raise ValueError(f"Asset {Fore.RED}'{name}'{Fore.RESET} not found.")
            # Update the asset
            SQL.update_asset(self.db._conn, cursor, name, image, mesh, description)
            logger.success(
                f"Asset {Fore.GREEN}'{name}'{Fore.RESET} updated successfully."
            )
        except ValueError as ve:
            raise
        except Exception as e:
            logger.error(f"Failed to update the asset '{name}': {e}")
            raise
