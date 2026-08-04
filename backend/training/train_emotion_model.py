import tensorflow as tf
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from dataset import train_dataset, test_dataset

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from config import *

# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ==========================================
# Compute Class Weights
# ==========================================

labels = []

for _, y in train_dataset.unbatch():
    labels.append(int(y.numpy()))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights")
print(class_weights)

# ==========================================
# Load EfficientNet
# ==========================================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ==========================================
# Phase 1
# Train only classifier
# ==========================================

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x)

x = GlobalAveragePooling2D()(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

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

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\n==========================")
print("PHASE 1 TRAINING")
print("==========================\n")

history1 = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights
)

# ==========================================
# Phase 2
# Fine Tune EfficientNet
# ==========================================

print("\n==========================")
print("PHASE 2 FINE TUNING")
print("==========================\n")

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
    epochs=20,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights
)

print("\n=================================")
print("Training Complete!")
print("Best model saved successfully.")
print("=================================")