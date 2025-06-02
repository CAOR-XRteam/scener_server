import json
import os
import sys
import torch

from colorama import Fore
from langchain_core.tools import tool
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

    # model_id = "openai/whisper-large-v3-turbo"

    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    # )
    # model.to(device)

    # processor = AutoProcessor.from_pretrained(model_id)

    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     torch_dtype=torch_dtype,
    #     device=device,
    # )

    # result = pipe(path, return_timestamps=True)

    model = whisper.load_model("turbo", device=device)
    result = model.transcribe(
        path,
        no_speech_threshold=0.1,
        condition_on_previous_text=False,
        logprob_threshold=-1.00,
        without_timestamps=True,
    )

    logger.info(
        f"{Fore.GREEN}Speech to text conversion completed: {result["text"]}{Fore.RESET}"
    )

    for r in result["segment"]:
        if r["no_speech_prob"] >= 0.9:
            logger.info(f"Probably not a speech segment: {r}")

    return result["text"]
