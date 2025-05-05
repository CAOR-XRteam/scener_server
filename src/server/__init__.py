import server.websocket as ws
from beartype import beartype

server = None


@beartype
def start(host: str = "localhost", port: int = 8000):
    global server
    if server is None:
        server = ws.Server(host, port)
    server.start()
