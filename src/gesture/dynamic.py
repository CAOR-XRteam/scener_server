from gesture.utils import compute_rotation, compute_position
from collections import deque
import numpy as np
import time
import threading

""" WIP not good currently """

class Dynamic:
    """In charge of computing the relative hand displacment and rotation"""
    def __init__(self, max_len=50):
        self.list_pose = deque(maxlen=max_len)
        self.list_rotation = deque(maxlen=max_len)
        self.lock = threading.Lock()
        self.displacement = None
        self.rotation = None
        self.old_pose = None
        self.old_rotation = None
        self.start_displacement_loop()

    def add_pose(self, landmarks):
        """Compute and store hand pose and rotation"""
        pose = compute_position(landmarks)
        rotation = compute_rotation(landmarks)

        with self.lock:
            self.list_pose.append(pose)
            self.list_rotation.append(rotation)

    def update_displacement(self, threshold=0.02):
        with self.lock:
            if not self.list_pose:
                return
            curr_pose = np.array(self.list_pose[-1])
            if self.old_pose is not None:
                delta = curr_pose - self.old_pose
                mask = np.abs(delta) <= threshold
                delta[mask] = 0  # on met à zéro les axes "trop petits"
                self.displacement = delta
            self.old_pose = curr_pose

    def update_relative_rotation(self, threshold=1.0):
        with self.lock:
            if not self.list_rotation:
                return
            curr_rotation = np.array(self.list_rotation[-1])
            if self.old_rotation is not None:
                delta = curr_rotation - self.old_rotation
                mask = np.abs(delta) <= threshold
                delta[mask] = 0  # on met à zéro les axes "trop petits"
                self.rotation = delta
            self.old_rotation = curr_rotation

    def start_displacement_loop(self, interval=0.1):
        def loop():
            while True:
                self.update_displacement()
                self.update_relative_rotation()
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def get_displacement(self):
        with self.lock:
            return self.displacement

    def get_rotation(self):
        with self.lock:
            return self.rotation
