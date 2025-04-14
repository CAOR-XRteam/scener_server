import json

def is_json(string):
    try:
        json.loads(string)  # Try to parse the string as JSON
        return True  # If parsing succeeds, it's a valid JSON
    except json.JSONDecodeError:
        return False  # If an error occurs, it's not a valid JSON
def convert(string):
    try:
        j = json.loads(string)  # Try to parse the string as JSON
        return j  # If parsing succeeds, it's a valid JSON
    except json.JSONDecodeError:
        return ""  # If an error occurs, it's not a valid JSON
