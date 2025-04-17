import json
import logging
from beartype import beartype

from ...lib import chat_call, deserialize_from_response_content

logger = logging.getLogger(__name__)


class SceneAnalyzer:
    @beartype
    def __init__(self, model_name):
        self.system_prompt = 'You are an assistant that analyzes a user\'s request to modify a 3D scene in the context of the current scene state. Your goal is to identify the necessary context from the scene to understand the request and determine if it\'s feasible or requires clarification. You will receive the current scene state (JSON) and the user\'s prompt (text) and you need to return a JSON in the same format with a relevant summary of the scene according to the user\'s prompt. Example Input: {"objects": [{"name": "table", "position": [0,0,0]}, {"name": "chair", "position": [1,0,1]}]} User: "Add a lamp on the table" Example Output: "{"objects": [{"name": "table", "position": [0,0,0]}]}"'
        self.model_name = model_name
        logger.info(f"Initialized with model: {self.model_name}")

    @beartype
    def analyze(self, current_scene: dict, user_input: str) -> dict:
        prompt = f"Current scene: {json.dumps(current_scene)}\nUser: {user_input}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        return deserialize_from_response_content(
            chat_call(self.model_name, messages, logger), logger
        )


if __name__ == "__main__":
    analyzer = SceneAnalyzer()
    print(
        analyzer.analyze(
            {"objects": [{"name": "table", "position": [0, 0, 0]}]},
            "Add a lamp on the table",
        )
    )
