from .api import *


def main(path):
    db.fill(path)
    db.read()

if __name__ == "__main__":
    path = "../media/asset"
    main(path)
