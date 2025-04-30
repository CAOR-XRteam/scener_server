from .database import DB

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
