import library.database as database


path = "../media/database.db"
db = None

def init():
    global db
    if db is None:
        db = database.DB(path)
    return db
