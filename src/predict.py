from transformers import TFAutoModel, AutoTokenizer
from utils import load_model, LABELS, BASE_MODEL
import numpy as np


def predict(input_text):
    tokenized_text = tokenizer(
        input_text, padding=True, truncation=True, return_tensors="tf"
    )
    predicts = model(tokenized_text)
    best_prediction = np.argmax(predicts)

    return LABELS[best_prediction]


model = load_model()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if __name__ == "__main__":

    while True:
        input_text = input("Enter text: ")
        if input_text.lower() == "exit":
            break
        print(predict(input_text))
