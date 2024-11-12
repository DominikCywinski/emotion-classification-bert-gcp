import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict


app = FastAPI()


class TextInput(BaseModel):
    text: str


@app.post("/predict/")
async def predict_emotion(input_data: TextInput):
    try:
        text = input_data.text
        prediction = predict(text)

        return {"text": text, "emotion": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
