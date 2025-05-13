"""
server.py

Handles websocket server connection and high-level functions.

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""

import asyncio
import signal
import websockets

from beartype import beartype
from colorama import Fore
from lib import logger
from server.client import Client


@beartype
class Server:
    """Manage server start / stop and handle clients"""

    # Main function
    def __init__(self, host: str, port: int):
        """Initialize server parameters"""
        self.host = host
        self.port = port
        self.list_client: list[Client] = []
        self.queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.server = None

    def start(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.handler_shutdown)
        try:
            loop.run_until_complete(self.run())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught in start(). Initiating shutdown.")
            if not self.shutdown_event.is_set():
                loop.run_until_complete(self.handler_shutdown(signal.SIGINT))
        except Exception as e:
            logger.error(f"Error in server's main execution: {e}")
        finally:
            if not self.server.is_serving():
                loop.run_until_complete(self.server.wait_closed())
            logger.info("Server finished working.")

    # Subfunction
    def handler_shutdown(self):
        """Manage Ctrl+C input to gracefully stop the server."""
        asyncio.create_task(self.shutdown())

    async def run(self):
        """Run the WebSocket server."""
        try:
            self.server = await websockets.serve(
                self.handler_client, "0.0.0.0", self.port
            )
            logger.info(
                f"Server running on {Fore.GREEN}ws://{self.host}:{self.port}{Fore.GREEN}"
            )
            print("---------------------------------------------")
            await self.server.wait_closed()
        except OSError as e:
            logger.error(
                f"Could not start server on {Fore.GREEN}ws://{self.host}:{self.port}{Fore.GREEN}: {e}."
            )
            self.shutdown_event.set()
        except Exception as e:
            logger.error(f"Internal error during server run: {e}")
            self.shutdown_event.set()

    async def handler_client(self, websocket):
        import server.client

        """Handle an incoming WebSocket client connection."""
        # Add client
        try:
            try:
                client = server.client.Client(websocket)
            except Exception as e:
                logger.error(
                    f"Failed to instantiate client for {websocket.remote_address}: {e}"
                )
                if websocket.open:
                    await websocket.close()
                return
            try:
                client.start()
            except Exception as e:
                logger.error(
                    f"Error starting client for {websocket.remote_address}: {e}"
                )
                if client in self.list_client:
                    self.list_client.remove(client)
                try:
                    await client.close()
                except Exception as exc:
                    logger.error(
                        f"Error closing client {websocket.remote_address} after start failure: {exc}",
                    )
                if websocket.open:
                    await websocket.close()
                return
            self.list_client.append(client)
            logger.info(f"New client connected:: {websocket.remote_address}")

            # Wait for the client to disconnect by awaiting the event
            try:
                await client.disconnection.wait()
            except asyncio.CancelledError:
                logger.error(
                    f"Task cancelled for client {websocket.remote_address} disconnection event."
                )
                if client.is_active:
                    asyncio.create_task(client.close())
        finally:
            # Perform necessary cleanup after client is disconnected
            logger.info(f"Client disconnected:: {websocket.remote_address}")
            self.list_client.remove(client)

    async def _close_client(self, client):
        try:
            await client.close()
        except Exception as e:
            logger.error(
                f"Error closing client {client.websocket.remote_address}: {e}",
            )
            client.is_active = False
            client.disconnection.set()
        finally:
            if client in self.list_client:
                self.list_client.remove(client)

    async def shutdown(self):
        """Gracefully shut down the server."""
        if self.server:
            self.server.close()
            try:
                await self.server.wait_closed()
                print("---------------------------------------------")
                logger.success(f"Server shutdown")
            except asyncio.CancelledError:
                logger.error(f"Server shutdown task cancelled")
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")

        # Filter out inactive clients and then close them
        for client in list(self.list_client):
            if client.is_active:
                # await client.close()
                self._close_client(client)

        self.list_client.clear()

        logger.info("All client connections processed for shutdown.")
        print("---------------------------------------------")
        logger.success(f"Server shutdown sequence completed.{Style.RESET_ALL}")
