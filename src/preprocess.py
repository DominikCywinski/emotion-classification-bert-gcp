import tensorflow as tf
from transformers import AutoTokenizer
from utils import load_my_dataset, MAX_SEQ_LENGTH, BATCH_SIZE, DATASET_NAME, BASE_MODEL


def tokenize(batch):
    return tokenizer(
        batch["text"], padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
    )


def order(input):
    data = list(input.values())

    return {
        "input_ids": data[1],
        "attention_mask": data[2],
        "token_type_ids": data[3],
    }, data[0]


def preprocess_dataset(dataset_name: str = DATASET_NAME, batch_size: int = BATCH_SIZE):
    print("Data preprocessing...")

    dataset = load_my_dataset(dataset_name)

    data_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    data_encoded.set_format(
        "tf", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(data_encoded["train"][:])
    train_dataset = train_dataset.batch(batch_size).shuffle(1000)
    train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(data_encoded["test"][:])
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

    return train_dataset, test_dataset


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
