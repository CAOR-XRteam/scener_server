import json
from ollama import chat


class SceneAnalyzer:
    def __init__(self):
        self.system_prompt = 'You are an assistant that analyzes a user\'s request to modify a 3D scene in the context of the current scene state. Your goal is to identify the necessary context from the scene to understand the request and determine if it\'s feasible or requires clarification. You will receive the current scene state (JSON) and the user\'s prompt (text) and you need to return a JSON in the same format with a relevant summary of the scene according to the user\'s prompt. Example Input: {"objects": [{"name": "table", "position": [0,0,0]}, {"name": "chair", "position": [1,0,1]}]} User: "Add a lamp on the table" Example Output: "{"objects": [{"name": "table", "position": [0,0,0]}]}"'
        self.model_name = "llama3.2"

    def analyze(self, current_scene: dict, user_input: str) -> dict:
        prompt = f"Current scene: {json.dumps(current_scene)}\nUser: {user_input}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = chat(self.model_name, messages)

        try:
            summary = json.loads(response.message.content)
        except (json.JSONDecodeError, ValueError) as e:
            summary = {"error": f"Invalid JSON: {str(e)}", "raw_response": response}

        return summary
