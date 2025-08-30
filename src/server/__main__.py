from server.api import ServerAPI
import os

if __name__ == "__main__":
    os.system("clear")
    server = ServerAPI()
    server.start()
