from library.manager.database import Database
import inspect
import os


# Path definition
path_current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
path_db = path_current + "/../../media/database.db"
path_asset = path_current + "/../../media/asset/"

# Database instance
db = Database(path_db)
