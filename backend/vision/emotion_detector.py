import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("backend/models/best_model.keras")

print("Emotion model loaded successfully!")