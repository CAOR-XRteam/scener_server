from library.library_database import Database
from library.library_list import Library
from library.library_asset import Asset
from lib import logger
import sys
import os
import inspect


class LibraryAPI:
    def __init__(self, db: Database):
        self.db = db
        self.library = Library(db)
        self.asset = Asset(db)

    def fill(self, path):
        """Fill the database with assets from the specified directory."""
        self.library.fill(path)

    def read(self):
        """Print out all the assets in the database."""
        return self.library.read()

    def get_list(self):
        """Return a list of all assets as dictionaries."""
        return self.library.get_list()

    def add_asset(self, name, image=None, mesh=None, description=None):
        """Add a new asset to the database."""
        self.asset.add(name, image, mesh, description)

    def update_asset(self, name, image=None, mesh=None, description=None):
        """Update an existing asset."""
        self.asset.update(name, image, mesh, description)

    def delete_asset(self, name):
        """Delete an asset by its name."""
        self.asset.delete(self, name)

    def list_asset():
        """Return a list of all assets as dictionaries."""
        return library.get_list()
