import server.websocket as ws

server = None


def start():
    global server
    if server is None:
        server = ws.Server()
    server.start()
