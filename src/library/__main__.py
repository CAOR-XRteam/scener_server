from .api import *


def main(path):
    db.fill(path)
    db.read()
    list = db.get_list()
    print(list)
    
if __name__ == "__main__":
    path = "../media/asset"
    main(path)
