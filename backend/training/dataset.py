import tensorflow as tf
from config import *

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasets/fer2013/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasets/fer2013/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training dataset loaded!")
print("Testing dataset loaded!")
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)