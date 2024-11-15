from transformers import TFAutoModel, AutoTokenizer
import numpy as np
import os
import tensorflow as tf
from preprocess import preprocess_dataset
from utils import BATCH_SIZE, LABELS, DATASET_NAME, BASE_MODEL, NEW_MODEL_PATH

np.random.seed(42)
tf.random.set_seed(42)


def train_model():
    print("Training model...")

    from model import BertClassification

    classifier = BertClassification(base_model, num_classes=len(LABELS))

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    classifier.fit(train_dataset, epochs=5)

    return classifier


train_dataset, _ = preprocess_dataset(dataset_name=DATASET_NAME, batch_size=BATCH_SIZE)

base_model = TFAutoModel.from_pretrained(BASE_MODEL)
classifier = train_model()

print("Saving weights...")
classifier.save_weights(NEW_MODEL_PATH, save_format="h5")
