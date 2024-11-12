import tensorflow as tf
import os
from datasets import load_dataset

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64
DATASET_NAME = "SetFit/emotion"
BASE_MODEL = "bert-base-uncased"
NEW_MODEL_NAME = "classifier.h5"

LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def load_my_dataset(dataset_name: str = DATASET_NAME):
    return load_dataset(dataset_name)


def load_model(base_model, model_name: str = "classifier.h5"):
    from model import BertClassification

    model = BertClassification(base_model, num_classes=len(LABELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    dummy_data = tf.zeros((1, MAX_SEQ_LENGTH), dtype=tf.int32)
    model(dummy_data)
    model.load_weights(os.path.join("models", model_name))

    return model
