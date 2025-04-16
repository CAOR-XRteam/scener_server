import json


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
