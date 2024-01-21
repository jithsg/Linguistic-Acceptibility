from fastapi import FastAPI, HTTPException
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

# Assuming DataModule is properly defined in data.py
from data import DataModule
from inference_onnx import ColaONNXPredictor


app = FastAPI()

# Initialize the predictor object
try:
    predictor = ColaONNXPredictor("./models/model.onnx")
except Exception as e:
    print(f"Failed to load the ONNX model: {e}")
    raise

@app.get("/")
async def read_root():
    return {"message": "Welcome to the CoLA ONNX Predictor API"}

@app.post("/predict")
async def get_prediction(text: str):
    try:
        result = predictor.predict(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# Optionally add more endpoints if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)