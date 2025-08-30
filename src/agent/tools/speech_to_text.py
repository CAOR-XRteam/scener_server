import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from colorama import Fore
from loguru import logger


@tool
def speech_to_texte(path: str) -> str:
    """Convert a vocal speech to text."""

    logger.info(f"Using tool {Fore.GREEN}{'speech_to_texte'}{Fore.RESET}")
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
    #print(result["text"])
    return result
