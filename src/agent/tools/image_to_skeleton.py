from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from colorama import Fore
from loguru import logger
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import cv2


@tool
def image_to_skeleton(path: str) -> str:
    """Convert an image to a skeleton"""

    logger.info(f"Using tool {Fore.GREEN}{'image_to_skeleton'}{Fore.RESET}")
    # Convert image to RGB and wrap as MediaPipe image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run gesture recognizer
    result = recognizer.recognize(mp_image)

    # Get top gesture
    if result.gestures:
        top_gesture = result.gestures[0][0]
        gesture_text = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
