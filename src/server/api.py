import server.server as ws


class ServerAPI:
    """API for the WebSocket server."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = ws.Server(host, port)

    def start(self):
        """Start the WebSocket server."""
        self.server.start()
