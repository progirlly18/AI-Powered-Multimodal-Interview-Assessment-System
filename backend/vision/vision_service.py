import cv2
import time
from collections import Counter

from backend.vision.emotion_detector import detect_emotion
from backend.vision.eye_contact import detect_eye_contact
from backend.vision.head_pose import detect_head_pose


def analyze_interview(duration=20):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    emotions = []
    eyes = []
    heads = []

    frame_count = 0
    start_time = time.time()

    while time.time() - start_time < duration:

        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        emotion, _, _ = detect_emotion(frame)
        eye = detect_eye_contact(frame)
        head = detect_head_pose(frame)

        emotions.append(emotion)
        eyes.append(eye)
        heads.append(head)

    cap.release()

    if len(emotions) == 0:

        return {
            "emotion": "No Face",
            "eye": "No Face",
            "head": "No Face"
        }

    return {

        "emotion": Counter(emotions).most_common(1)[0][0],

        "eye": Counter(eyes).most_common(1)[0][0],

        "head": Counter(heads).most_common(1)[0][0]

    }