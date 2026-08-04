import tensorflow as tf
from config import *

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasets/fer2013/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(train_dataset.class_names)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasets/fer2013/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

from tensorflow.keras.applications.efficientnet import preprocess_input

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("Training dataset loaded!")
print("Testing dataset loaded!")