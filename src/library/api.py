from .database import DB
from loguru import logger
import sys


#Loguru config
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

path = "../media/database.db"
db = DB(path)

def add_asset():
    pass

def update_asset():
    pass

def remove_asset():
    pass

def list_asset():
    return db.get_list()
