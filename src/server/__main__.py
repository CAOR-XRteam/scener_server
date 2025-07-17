from server.api import ServerAPI
import os


if __name__ == "__main__":
    os.system("clear")
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", 8765))
    server = ServerAPI(host=HOST, port=PORT)
    server.start()
