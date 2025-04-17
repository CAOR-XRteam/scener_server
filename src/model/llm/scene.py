import json
import logging
from beartype import beartype
from ollama import chat


class SceneAnalyzer:
    @beartype
    def __init__(self, model_name: str = "llama3.2"):
        self.system_prompt = 'You are an assistant that analyzes a user\'s request to modify a 3D scene in the context of the current scene state. Your goal is to identify the necessary context from the scene to understand the request and determine if it\'s feasible or requires clarification. You will receive the current scene state (JSON) and the user\'s prompt (text) and you need to return a JSON in the same format with a relevant summary of the scene according to the user\'s prompt. Example Input: {"objects": [{"name": "table", "position": [0,0,0]}, {"name": "chair", "position": [1,0,1]}]} User: "Add a lamp on the table" Example Output: "{"objects": [{"name": "table", "position": [0,0,0]}]}"'
        self.model_name = model_name
        logging.info(f"SceneAnalyzer initialized with model: {self.model_name}")

    @beartype
    def analyze(self, current_scene: dict, user_input: str) -> dict:
        prompt = f"Current scene: {json.dumps(current_scene)}\nUser: {user_input}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = chat(self.model_name, messages)
        except Exception as e:
            logging.error(f"Error during SceneAnalyzer chat API call: {str(e)}")
            return {"error": f"SceneAnalyzer chat API call failed: {str(e)}"}

        try:
            summary = json.loads(response.message.content)
        except (json.JSONDecodeError, ValueError) as e:
            summary = {
                "error": f"Invalid JSON: {str(e)}",
                "raw_response": response.message.content if response else None,
            }

        logging.info(f"SceneAnalyzer chat API call successful: {summary}")
        return summary


if __name__ == "__main__":
    analyzer = SceneAnalyzer()
    try:
        result = analyzer.analyze(
            {"objects": [{"name": "table", "position": [0, 0, 0]}]},
            "Add a lamp on the table",
        )
        print(result)
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
