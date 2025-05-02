from .database import DB
from .library import Library
from loguru import logger
import sys


#Loguru config
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

path = "../media/database.db"
db = DB(path)
library = Library(db)


def add_asset():
    pass

def update_asset():
    pass

def remove_asset():
    pass

def list_asset():
    return asset_manager.get_list()
