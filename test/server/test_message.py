import pytest
import asyncio
import json
from websockets import connect
from client import send_message


@pytest.mark.asyncio
async def test_json():
    uri = "ws://localhost:8765"

    post = {"message": "Hello !"}

    response = await send_message(uri, json.dumps(post))
    response_data = json.loads(response)

    assert response_data["message"] == "No command in the json"
    assert response_data["status"] == "error"
    assert response_data["code"] == 400


@pytest.mark.asyncio
async def test_non_json():
    uri = "ws://localhost:8765"

    response = await send_message(uri, "hello !")
    response_data = json.loads(response)

    assert response_data["message"] == "Message not in JSON format"
    assert response_data["status"] == "error"
    assert response_data["code"] == 400


@pytest.mark.asyncio
async def test_empty():
    uri = "ws://localhost:8765"

    response = await send_message(uri, "")
    response_data = json.loads(response)

    assert response_data["message"] == "Empty message received"
    assert response_data["status"] == "error"
    assert response_data["code"] == 400


@pytest.mark.asyncio
async def test_llm():
    uri = "ws://localhost:8765"

    post = {"command": "chat", "message": "answer by saying only 'Hello !'"}

    response = await send_message(uri, json.dumps(post))
    response_data = json.loads(response)

    assert response_data["message"] == "Hello !"
    assert response_data["status"] == "success"
    assert response_data["code"] == 200

    response = await send_message(uri, json.dumps(post))
    response_data = json.loads(response)

    assert response_data["message"] == "Hello !"
    assert response_data["status"] == "success"
    assert response_data["code"] == 200
