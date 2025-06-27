from gesture.dynamic import Dynamic
from gesture.hand import Hand
from gesture.image import crop_hand
from gesture.utils import compute_rotation, compute_position
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import cv2
import time
import threading


class Mediapipe:
    # Main function
    def __init__(self):
        """Initialize parameters"""

        # Gesture Recognizer setup
        base_options = python.BaseOptions(
            model_asset_path="model/mediapipe/gesture_recognizer.task",
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        )
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Initialize drawing utils and solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Instantiate hand detection
        self.hands_skeleton = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        self.hand_right = Hand("Right")
        self.hand_left = Hand("Left")

    # Processing draw_hand_stuff
    def process_skeleton(self, frame):
        """Recognize gesture in an image as input"""
        result = self.hands_skeleton.process(frame)
        return result

    def process_hand(self, skeletons, frame, hand):
        # Find hand index
        for i, handedness in enumerate(skeletons.multi_handedness):
            if handedness.classification[0].label == hand.label:
                hand.index = i
                break
        if hand.index is None:
            return

        # Fill hand stuff
        if (
            hand.index is not None
            and len(skeletons.multi_hand_landmarks) >= hand.index + 1
        ):
            hand.landmarks = skeletons.multi_hand_landmarks[hand.index]
            hand.image = crop_hand(frame, hand.landmarks)
            self.process_gesture(hand)

    def process_gesture(self, hand):
        """Recognize gesture in an image as input"""
        rgb_frame = cv2.cvtColor(hand.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res_gesture = self.recognizer.recognize(mp_image)
        if res_gesture.gestures:
            hand.gesture = res_gesture.gestures[0][0].category_name
            hand.score = res_gesture.gestures[0][0].score

    # Drawing stuff
    def draw_result(self, frame):
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        self.draw_hand_stuff(bw, self.hand_left)
        self.draw_hand_stuff(bw, self.hand_right)
        cv2.imshow("Hand gesture", bw)

    def draw_hand_stuff(self, frame, hand):
        # Draw hand landmarks
        if hand.landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                hand.landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Draw hand label
            wrist = hand.landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            score_text = f"{hand.score:.2f}" if hand.score is not None else "N/A"
            shape = frame.shape
            x = int(wrist.x * shape[1])
            y = int(wrist.y * shape[0])
            cv2.putText(
                frame,
                hand.label,
                (x - 100, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                hand.gesture,
                (x - 100, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                score_text,
                (x - 100, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def test(self):
        """Test with webcam"""
        # Webcam feed
        cap = cv2.VideoCapture(0)
        dyn = Dynamic()

        while cap.isOpened():
            start = time.time()

            # New frame
            ret, frame = cap.read()
            if not ret:
                break
            self.hand_left.reset()
            self.hand_right.reset()

            # Hand skeletons
            skeletons = self.process_skeleton(frame)
            if skeletons.multi_hand_landmarks:
                self.process_hand(skeletons, frame, self.hand_left)
                self.process_hand(skeletons, frame, self.hand_right)

            # Show frame
            self.draw_result(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            end = time.time()
            print(f"Execution time: {(end - start)*1000:.2f} ms")

        cap.release()
        cv2.destroyAllWindows()
        self.hands_skeleton.close()
