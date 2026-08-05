import cv2

from backend.vision.emotion_detector import detect_emotion
from backend.vision.eye_contact import detect_eye_contact
from backend.vision.head_pose import detect_head_pose


class VisionPipeline:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

    def analyze(self):

        ret, frame = self.cap.read()

        if not ret:

            return None

        emotion, confidence, bbox = detect_emotion(frame)

        eye = detect_eye_contact(frame)

        head = detect_head_pose(frame)

        return {

            "emotion": emotion,

            "emotion_confidence": confidence,

            "eye": eye,

            "head": head,

            "frame": frame,

            "bbox": bbox

        }

    def release(self):

        self.cap.release()