from library import db
from library.api import LibraryAPI
from library.manager.database import Database
import inspect
import os


if __name__ == "__main__":
    """ test the library with root media fodler """

    # Read database
    api = LibraryAPI(db)
    api.read()
    list = api.get_list()
    print(list)
