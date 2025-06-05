from gesture.dynamic import Dynamic
from gesture.utils import compute_rotation, compute_position
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import mediapipe as mp
import numpy as np
import cv2

class Hand:
    # Main function
    def __init__(self, label):
        """Initialize parameters"""
        self.index = None
        self.image = None
        self.label = label

        self.pose = None
        self.rotation = None
        self.landmarks = None
        self.gesture = None
        self.score = None

        self.list_pose = deque(maxlen=50)
        self.list_rotation = deque(maxlen=50)
        self.dyn = Dynamic()

        if label not in ("Right", "Left"):
            raise ValueError("name must be 'Right' or 'Left'")

    def reset(self):
        self.index = None
        self.image = None
        self.pose = None
        self.rotation = None
        self.landmarks = None
        self.gesture = None
        self.score = None
