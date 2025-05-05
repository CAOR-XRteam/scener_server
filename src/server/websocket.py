import asyncio
import signal
import websockets
import logging

from beartype import beartype
from colorama import Fore

import utils


logger = logging.getLogger(__name__)


@beartype
class Server:
    """Manage server start / stop and handle clients"""

    # Main function
    def __init__(self, host: str, port: str, model_name: str = "llama3.1"):
        """Initialize server parameters"""
        self.host = host
        self.port = port
        self.model_name = model_name
        self.list_client = []
        self.queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.server = None

    def start(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.handler_shutdown)
        loop.run_until_complete(self.run())

    # Subfunction
    def handler_shutdown(self):
        """Manage Ctrl+C input to gracefully stop the server."""
        asyncio.create_task(self.shutdown())

    async def run(self):
        """Run the WebSocket server."""
        self.server = await websockets.serve(
            self.handler_client, self.host, self.port, self.model_name
        )
        logger.info(
            f"Server running on {Fore.GREEN}ws://{self.host}:{self.port}{Fore.GREEN}"
        )
        print("---------------------------------------------")
        await self.server.wait_closed()

    async def handler_client(self, websocket, model_name):
        import server.client

        """Handle an incoming WebSocket client connection."""
        # Add client
        client = server.client.Client(websocket, model_name)
        client.start()
        self.list_client.append(client)
        logger.info(f"New client connected:: {websocket.remote_address}")

        # Wait for the client to disconnect by awaiting the event
        await client.disconnection.wait()

        # Perform necessary cleanup after client is disconnected
        logger.info(f"Client disconnected:: {websocket.remote_address}")
        self.list_client.remove(client)

    async def shutdown(self):
        """Gracefully shut down the server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("---------------------------------------------")
            logger.success(f"Server shutdown")

        # Filter out inactive clients and then close them
        for client in list(self.list_client):
            if client.is_active:
                await client.close()

        self.list_client.clear()
