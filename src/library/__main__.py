from library.api import LibraryAPI
import inspect
import os
from library.library_database import Database

if __name__ == "__main__":
    """ test the library with root media fodler """
    db_path = (
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        + "/../../media/database.db"
    )
    db = Database(db_path)
    LibraryAPI = LibraryAPI(db)

    assets_path = (
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        + "/../../media/asset/"
    )

    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    LibraryAPI.fill(assets_path)
    LibraryAPI.read()
    list = LibraryAPI.get_list()
    print(list)
