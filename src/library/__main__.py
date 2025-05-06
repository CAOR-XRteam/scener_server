from library.api import *


def main(path):
    library.fill(path)
    library.read()
    list = library.get_list()
    print(list)


if __name__ == "__main__":
    path = "/home/artem/Scener/src/media/asset"
    main(path)
