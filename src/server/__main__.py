from api import *
import os


def start():
    global server
    if server is None:
        server = ws.Server()
    server.start()

def main():
    server.start()

if __name__ == "__main__":
    os.system("clear")
    main()
