from beartype import beartype
from lib import logger
from library import db

from library.manager.asset import Asset
from library.manager.library import AppAsset, AssetFinder, NullableAppAsset
from library.manager.library import Library

# TODO: more precise error handling to propagate to the agent, tests


@beartype
class LibraryAPI:
    def __init__(self):
        self.db = db
        self.library = Library(db)
        self.asset = Asset(db)

    def fill(self, path):
        """Fill the database with assets from the specified directory."""
        try:
            self.library.fill(path)
        except Exception as e:
            logger.error(f"Failed to fill the database: {e}")
            raise

    def read(self):
        """Print out all the assets in the database."""
        try:
            return self.library.read()
        except Exception as e:
            logger.error(f"Failed to read the database: {e}")
            raise

    def get_list(self):
        """Return a list of all assets as dictionaries."""
        try:
            return self.library.get_list()
        except Exception as e:
            logger.error(f"Failed to get the list of assets: {e}")
            raise

    def add_asset(self, name, image=None, mesh=None, description=None):
        """Add a new asset to the database."""
        try:
            self.asset.add(name, image, mesh, description)
        except Exception as e:
            logger.error(f"Failed to add asset: {e}")
            raise

    def update_asset(self, name, image=None, mesh=None, description=None):
        """Update an existing asset."""
        try:
            self.asset.update(name, image, mesh, description)
        except Exception as e:
            logger.error(f"Failed to update asset: {e}")
            raise

    def delete_asset(self, name):
        """Delete an asset by its name."""
        try:
            self.asset.delete(name)
        except Exception as e:
            logger.error(f"Failed to delete asset: {e}")
            raise

    def get_asset(self, name):
        """Get an asset by its name"""
        try:
            self.library.get_asset(name)
        except Exception as e:
            logger.error(f"Failed to get asset: {e}")
            raise

    def find_asset_by_description(self, description: str) -> NullableAppAsset:
        """Find the closest asset to a given description"""
        asset_finder = AssetFinder(self.library.get_list())
        try:
            return asset_finder.find_by_description(description)
        except Exception:
            raise
