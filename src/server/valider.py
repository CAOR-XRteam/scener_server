import json
import io

from pydantic import BaseModel, field_validator
from typing import Literal


def validate_message(m):
    if not m or m.isspace():
        raise ValueError("Message must not be empty or whitespace")
    return m


def parse_agent_response(m: str):
    thinking, final_aswer = m.split("Final answer: ")
    return thinking, final_aswer


def convert_image_to_bytes(image_path):
    try:
        with open(image_path, "rb") as image:
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            return byte_arr.getvalue()
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        raise


class OutputMessage(BaseModel):
    status: Literal["stream", "error"]
    code: int
    action: Literal["agent_response", "image_generation", "thinking_process"]
    message: str
    data: bytes = (None,)
    _validate_message = field_validator("message")(validate_message)


class InputMessage(BaseModel):
    command: Literal["chat"]
    message: str

    _validate_message = field_validator("message")(validate_message)


# Main function
async def check_message(client, message):
    """Handle the incoming message and return the response."""
    # Check validity
    if not message:
        await client.send_message("error", 400, "Empty message received")
        return False

    if not is_json(message):
        await client.send_message("error", 400, "Message not in JSON format")
        return False

    if not has_command(message):
        await client.send_message("error", 400, "No command in the json")
        return False

    # Message is valid
    return True


# Subfunction
def is_json(message):
    """Check if the message is a valid JSON."""
    try:
        json.loads(message)  # Try to parse the message as JSON
        return True
    except json.JSONDecodeError:
        return False


def has_command(message):
    """Check if the message contains a 'command' field."""
    try:
        j = json.loads(message)  # Parse the JSON
        if "command" in j:
            return True
        else:
            return False
    except json.JSONDecodeError:
        return False
