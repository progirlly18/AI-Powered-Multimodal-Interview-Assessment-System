from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import cv2

# Load model
model = load_model("backend/models/best_model.keras")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# FaceMesh detector
detector = FaceMeshDetector(maxFaces=1)


def detect_emotion(frame):

    img, faces = detector.findFaceMesh(frame, draw=False)

    if not faces:
        return "No Face", 0.0, None

    face = faces[0]

    xs = [p[0] for p in face]
    ys = [p[1] for p in face]

    x1 = max(0, min(xs) - 20)
    y1 = max(0, min(ys) - 20)
    x2 = min(frame.shape[1], max(xs) + 20)
    y2 = min(frame.shape[0], max(ys) + 20)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "No Face", 0.0, None

    crop = cv2.resize(crop, (224, 224))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    crop = crop.astype("float32")
    crop = preprocess_input(crop)

    crop = np.expand_dims(crop, axis=0)

    pred = model.predict(crop, verbose=0)[0]

    idx = np.argmax(pred)

    confidence = float(pred[idx])

    emotion = emotion_labels[idx]

    return emotion, confidence, (x1, y1, x2, y2)