import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict


def test_predict_emotion():
    input_text = "I feel happy today!"
    expected_emotion = "joy"

    result = predict(input_text)

    assert result == expected_emotion, f"Expected {expected_emotion}, but got {result}"
