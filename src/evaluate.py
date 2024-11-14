from preprocess import preprocess_dataset
from utils import (
    BATCH_SIZE,
    DATASET_NAME,
    load_model,
)

train_dataset, test_dataset = preprocess_dataset(
    dataset_name=DATASET_NAME, batch_size=BATCH_SIZE
)

model = load_model()

print("Evaluating model...")

print("Training set...")
model.evaluate(train_dataset)

print("Training set...")
model.evaluate(test_dataset)
