from transformers import TFAutoModel, AutoTokenizer
from utils import (
    BATCH_SIZE,
    DATASET_NAME,
    load_model,
    BASE_MODEL,
    NEW_MODEL_NAME,
)

from preprocess import preprocess_dataset


train_dataset, test_dataset = preprocess_dataset(
    dataset_name=DATASET_NAME, batch_size=BATCH_SIZE
)

base_model = TFAutoModel.from_pretrained(BASE_MODEL)

model = load_model(base_model=base_model, model_name=NEW_MODEL_NAME)

print("Evaluating model...")

print("Training set...")
model.evaluate(train_dataset)
print("Training set...")
model.evaluate(test_dataset)
