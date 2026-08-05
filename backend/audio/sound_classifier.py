import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np

print("Loading YAMNet model...")

model = hub.load("https://tfhub.dev/google/yamnet/1")

print("Model loaded!")

class_map_path = model.class_map_path().numpy().decode("utf-8")

with open(class_map_path) as f:
    labels = [line.strip().split(",")[2] for line in f.readlines()[1:]]

waveform, sr = librosa.load("sample.wav", sr=16000)

scores, embeddings, spectrogram = model(waveform)

scores = scores.numpy()

mean_scores = np.mean(scores, axis=0)

predicted = np.argmax(mean_scores)

label = labels[predicted]

confidence = float(mean_scores[predicted])

print("\nDetected Sound :", label)
print("Confidence     :", round(confidence,3))


# -------------------------
# Decision Engine
# -------------------------

NATURAL = [
    "Rain",
    "Thunder",
    "Wind",
    "Water",
    "Ocean",
    "Stream",
    "Bird",
    "Bird vocalization",
    "Insect",
    "Cricket",
    "Thunderstorm"
]

SUSPICIOUS = [
    "Speech",
    "Conversation",
    "Narration",
    "Television",
    "Music",
    "Typing",
    "Keyboard",
    "Mouse click",
    "Whispering"
]

print("\nDecision")

if any(x.lower() in label.lower() for x in NATURAL):

    print("✅ Natural Background Sound")
    print("No malpractice detected.")

elif any(x.lower() in label.lower() for x in SUSPICIOUS):

    print("🚨 Possible External Assistance")
    print("Review required.")

else:

    print("⚠ Unknown Background Sound")
    print("Manual review recommended.")