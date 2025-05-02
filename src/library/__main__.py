from .api import *
import os


def main(path):
    library.fill(path)
    library.read()
    list = library.get_list()
    print(list)

if __name__ == "__main__":
    path = "../media/asset"
    main(path)
