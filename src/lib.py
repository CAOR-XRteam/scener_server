import json
import os
import sys
import torch

from sdk.scene import Scene
from colorama import Fore
from loguru import logger
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def load_config():
    """Load the configuration from the JSON file."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file '{CONFIG_PATH}' not found.")

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> | {message}",
    backtrace=True,
)


def speech_to_text(path: str) -> str:
    """Convert a vocal speech to text."""
    import whisper

    logger.info(
        f"{Fore.YELLOW}Speech to text conversion started for file: {path}{Fore.RESET}"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(path, return_timestamps=True)

    logger.info(
        f"{Fore.GREEN}Speech to text conversion completed: {result["text"]}{Fore.RESET}"
    )

    return result["text"]


def deserialize_scene_json(scene_json: str) -> Scene:
    """Deserialize a JSON scene description into a Scene object."""
    try:
        scene_data = json.loads(scene_json)
        scene = Scene(**scene_data)
        logger.info(f"Scene deserialized successfully: {scene}")
        return scene
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        raise ValueError("Invalid JSON format for scene data.")
    except Exception as e:
        logger.error(f"Error deserializing scene: {e}")
        raise ValueError("Failed to deserialize scene data.")


if __name__ == "__main__":
    scene_dict = {
        "name": "Test Sun Scene",
        "timestamp": "2023-10-27T10:00:00Z",
        "skybox": {
            "type": "sun",
            "top_color": {"r": 0.2, "g": 0.4, "b": 0.8},
            "top_exponent": 1.0,
            "horizon_color": {"r": 0.9, "g": 0.8, "b": 0.6},
            "bottom_color": {"r": 0.3, "g": 0.3, "b": 0.35},
            "bottom_exponent": 1.0,
            "sky_intensity": 1.2,
            "sun_color": {"r": 1.0, "g": 0.9, "b": 0.8},
            "sun_intensity": 1.5,
            "sun_alpha": 20.0,
            "sun_beta": 20.0,
            "sun_vector": {"x": 0.5, "y": 0.5, "z": 0.0, "w": 0.0},
        },
        "graph": [
            {
                "id": "light1",
                "name": "Directional Light",
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 50, "y": -30, "z": 0},
                "scale": {"x": 1, "y": 1, "z": 1},
                "components": [
                    {
                        "componentType": "light",
                        "type": "directional",
                        "color": {"r": 1.0, "g": 0.95, "b": 0.85},
                        "intensity": 1.1,
                        "indirect_multiplier": 1.0,
                        "mode": "realtime",
                        "shadow_type": "soft_shadows",
                    }
                ],
            },
            {
                "id": "light2",
                "name": "Point Light Source",
                "position": {"x": 5, "y": 2, "z": 3},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "scale": {"x": 1, "y": 1, "z": 1},
                "components": [
                    {
                        "componentType": "light",
                        "type": "point",
                        "color": {"r": 1.0, "g": 0.5, "b": 0.2},
                        "intensity": 2.5,
                        "indirect_multiplier": 1.0,
                        "range": 15.0,
                        "mode": "mixed",
                        "shadow_type": "hard_shadows",
                    }
                ],
            },
            {
                "id": "obj1",
                "name": "GroundPlane",
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "scale": {"x": 10, "y": 1, "z": 10},
                "components": [
                    {
                        "componentType": "Primitive",
                        "shape": "plane",
                        "color": {"r": 0.5, "g": 0.5, "b": 0.5},
                    }
                ],
            },
            {
                "id": "obj2",
                "name": "MainCube",
                "position": {"x": 0, "y": 1, "z": 5},
                "rotation": {"x": 0, "y": 25, "z": 0},
                "scale": {"x": 2, "y": 2, "z": 2},
                "components": [
                    {
                        "componentType": "primitive",
                        "shape": "cube",
                        "color": {"r": 0.8, "g": 0.1, "b": 0.1},
                    }
                ],
            },
            {
                "id": "theatre",
                "name": "Theatre Container",
                "position": {"x": -5, "y": 0, "z": 10},
                "rotation": {"x": 0, "y": 180, "z": 0},
                "scale": {"x": 5, "y": 5, "z": 5},
                "components": [{"componentType": "dynamic", "id": "theatre"}],
            },
        ],
    }
    scene_json = json.dumps(scene_dict)
    print(f"Deserialized scene: {deserialize_scene_json(scene_json)}")
