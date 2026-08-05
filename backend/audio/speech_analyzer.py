import whisper
import tensorflow_hub as hub
import librosa
import numpy as np
import sounddevice as sd

from scipy.io.wavfile import write
from collections import Counter
import re

# -----------------------
# RECORD AUDIO
# -----------------------

fs = 16000
seconds = 10

print("Recording starts in 3 seconds...")

sd.sleep(3000)

print("Speak now...")

recording = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    dtype="int16"
)

sd.wait()

write("sample.wav", fs, recording)

print("Recording complete!")

# -----------------------
# SPEECH TO TEXT
# -----------------------

print("\nLoading Whisper...")

whisper_model = whisper.load_model("base")

result = whisper_model.transcribe("sample.wav")

transcript = result["text"]

print("\nTranscript")
print("--------------------------------")
print(transcript)

# -----------------------
# FILLER WORDS
# -----------------------

fillers = [
    "um",
    "uh",
    "like",
    "actually",
    "basically",
    "you know",
    "sort of",
    "kind of"
]

print("\nFiller Words")
print("--------------------------------")

total_fillers = 0

text = transcript.lower()

for filler in fillers:

    count = len(
        re.findall(
            r"\b" + re.escape(filler) + r"\b",
            text
        )
    )

    if count:

        print(f"{filler:<15}: {count}")

    total_fillers += count

print("--------------------------------")
print("Total Fillers:", total_fillers)

# -----------------------
# WORDS PER MINUTE
# -----------------------

word_count = len(text.split())

wpm = round(word_count / (seconds / 60))

print("\nSpeech Statistics")
print("--------------------------------")

print("Words:", word_count)

print("Duration:", seconds, "seconds")

print("Speaking Rate:", wpm, "WPM")

# -----------------------
# SOUND CLASSIFICATION
# -----------------------

print("\nLoading YAMNet...")

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

class_map = yamnet.class_map_path().numpy().decode("utf-8")

with open(class_map) as f:
    labels = [line.strip().split(",")[2] for line in f.readlines()[1:]]

waveform, sr = librosa.load("sample.wav", sr=16000)

scores, embeddings, spectrogram = yamnet(waveform)

scores = scores.numpy()

mean_scores = np.mean(scores, axis=0)

idx = np.argmax(mean_scores)

sound = labels[idx]

confidence = float(mean_scores[idx])

print("\nBackground Sound")
print("--------------------------------")

print(sound)

print("Confidence:", round(confidence,3))

# -----------------------
# DECISION
# -----------------------

NATURAL = [
    "Rain",
    "Thunder",
    "Wind",
    "Water",
    "Ocean",
    "Bird",
    "Stream"
]

SUSPICIOUS = [
    "Conversation",
    "Television",
    "Typing",
    "Keyboard",
    "Music"
]

print("\nEnvironment Decision")
print("--------------------------------")

if any(x.lower() in sound.lower() for x in NATURAL):

    print("Natural Background Noise")

elif any(x.lower() in sound.lower() for x in SUSPICIOUS):

    print("Possible External Assistance")

else:

    print("No suspicious background detected")

# -----------------------
# SPEECH SCORE
# -----------------------

score = 100

score -= total_fillers * 3

if wpm < 90:

    score -= 10

elif wpm > 170:

    score -= 10

score = max(0, score)

print("\nSpeech Score")
print("--------------------------------")

print(score, "/100")