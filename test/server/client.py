import asyncio
import websockets


async def send_message(uri, message):
    """Send a message to the WebSocket server and return the response."""
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)
        response = await websocket.recv()
        return response
