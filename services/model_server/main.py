from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import onnxruntime as ort
from typing import List

app = FastAPI()

MODELS_DIR = Path("models")

model_cache = {}

class PredictionRequest(BaseModel):
    model_name: str
    input: List[float]

class PredictionResponse(BaseModel):
    output: List[float]
    status: str

def load_model(model_name: str):
    """check if the models in cache or not"""
    if model_name in model_cache:
        return model_cache[model_name]

    model_path = MODELS_DIR / f"{model_name}.onnx"

    if not model_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found in models directory"
        )

    try:
        session = ort.InferenceSession(str(model_path))
        model_cache[model_name] = session
        return session
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@app.get("/")
async def root():
    return {"Status": "The server is up and running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

