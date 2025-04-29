from .api import *
import os


def main(path):
    db.fill(path)
    db.read()
    list = db.get_list()
    print(list)

if __name__ == "__main__":
    os.system("clear")
    path = "../media/asset"
    main(path)
