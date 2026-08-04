import tensorflow as tf
from dataset import train_dataset, test_dataset
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from config import *

# -----------------------------
# Data Augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# -----------------------------
# Load EfficientNet
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Phase 1
base_model.trainable = False

# -----------------------------
# Build Model
# -----------------------------
inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = base_model(x)

x = GlobalAveragePooling2D()(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# -----------------------------
# Compile Phase 1
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    filepath="backend/models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

print("\n========================")
print("PHASE 1 TRAINING")
print("========================\n")

history1 = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[checkpoint]
)

# -----------------------------
# Phase 2 Fine Tuning
# -----------------------------
print("\n========================")
print("PHASE 2 FINE TUNING")
print("========================\n")

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    callbacks=[checkpoint]
)

print("\nTraining Complete!")
