from transformers import TFAutoModel, AutoTokenizer
from utils import load_model, LABELS, BASE_MODEL, NEW_MODEL_NAME
import numpy as np


def predict(input_text):
    tokenized_text = tokenizer(
        input_text, padding=True, truncation=True, return_tensors="tf"
    )
    predicts = model(tokenized_text)
    best_prediction = np.argmax(predicts)

    return LABELS[best_prediction]


base_model = TFAutoModel.from_pretrained(BASE_MODEL)
model = load_model(base_model=base_model, model_name=NEW_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if __name__ == "__main__":

    while True:
        input_text = input("Enter text: ")
        if input_text.lower() == "exit":
            break
        print(predict(input_text))
