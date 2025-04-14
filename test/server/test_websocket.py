import pytest
import asyncio
import json
from websockets import connect
from client import send_message


@pytest.mark.asyncio
async def test_server_up():
    """Test if the WebSocket server is up and running before running any other tests."""
    uri = "ws://localhost:8765"

    try:
        async with connect(uri) as websocket:
            pass
    except Exception as e:
        pytest.fail(f"WebSocket server is not up or reachable: {e}")  # Skip the tests if server is down
