from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2

# Load trained model
model = load_model("backend/models/best_model.keras")

# Emotion labels (must match dataset folder order)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade XML file.")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Press 'q' to quit.\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # -----------------------------
        # Add margin around face
        # -----------------------------
        margin = 20

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Resize
        face = cv2.resize(face, (224, 224))

        # BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # EfficientNet preprocessing
        face = face.astype("float32")
        face = preprocess_input(face)

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Predict
        predictions = model.predict(face, verbose=0)[0]

        # Print probabilities
        print("\n------------------------------")
        for label, prob in zip(emotion_labels, predictions):
            print(f"{label:<10}: {prob:.4f}")

        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]

        if confidence < 0.45:
            emotion = "Uncertain"
        else:
            emotion = emotion_labels[predicted_index]

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display emotion
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Interview AI - Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()