from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
import tensorflow as tf
import numpy as np
from preprocess import preprocess

model = TFAutoModel.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
emotions = load_dataset("SetFit/emotion")


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded.set_format(
    "tf", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
)

BATCH_SIZE = 64


def order(input):
    data = list(input.values())
    return {
        "input_ids": data[1],
        "attention_mask": data[2],
        "token_type_ids": data[3],
    }, data[0]


train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded["train"][:])
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded["test"][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)


class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert_model = bert_model
        self.fc = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, inputs):
        x = self.bert_model(inputs)
        x = x.pooler_output
        return self.fc(x)


classifier = BERTForClassification(model, num_classes=6)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = classifier.fit(train_dataset, epochs=1)

classifier.evaluate(test_dataset)

emotions_dict = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

data = emotions["test"][0]
text = data["text"]

tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
preds = classifier(tokenized_text)
best_pred = np.argmax(preds)

# for i in range(len(emotions['test'])):
for i in range(10):

    data = emotions["test"][i]
    text = data["text"]
    y_true = data["label"]
    y_true_emotion = data["label_text"]
    tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
    preds = classifier(tokenized_text)
    best_pred = np.argmax(preds)

    print(f"{text} -> {y_true_emotion} -> {emotions_dict[best_pred]}")
