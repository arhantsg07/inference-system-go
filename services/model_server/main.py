import base64
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"

model_cache = {}

class PredictionRequest(BaseModel):
    model_name: str
    input: str

class PredictionResponse(BaseModel):
    model_name: str
    output: List[dict]
    status: str
    inference_time_ms: float

def load_model(model_name: str):
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

def preprocess_image(image_data: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[dict]:
    outputs = outputs[0][0]
    outputs = np.transpose(outputs, (1, 0))
    rows, _ = outputs.shape

    boxes, scores, class_ids = [], [], []
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.max(classes_scores)
        if max_score >= conf_threshold:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][:4]
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(640, int(x + w / 2))
            y2 = min(640, int(y + h / 2))
            boxes.append([x1, y1, x2, y2])
            scores.append(float(max_score))
            class_ids.append(int(class_id))

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    detections = []
    for i in indices:
        detections.append({
            "class_id": class_ids[i],
            "confidence": round(scores[i], 4),
            "bbox": boxes[i]
        })
    return detections

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        session = load_model(request.model_name)
        input_tensor = preprocess_image(request.input)
        input_name = session.get_inputs()[0].name

        import time
        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_tensor})
        inference_time = (time.perf_counter() - start) * 1000

        detections = postprocess(outputs)

        return PredictionResponse(
            model_name=request.model_name,
            output=detections,
            status="ok",
            inference_time_ms=round(inference_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/models")
async def models():
    if not MODELS_DIR.exists():
        return {"models": []}
    models_list = [f.stem for f in MODELS_DIR.glob("*.onnx")]
    return {"models": models_list}

@app.get("/model_info/{model_name}")
async def get_model_information(model_name: str):
    session = load_model(model_name)
    input_info = []
    for inp in session.get_inputs():
        input_info.append({
            "name": inp.name,
            "shape": inp.shape,
            "type": inp.type
        })
    output_info = []
    for out in session.get_outputs():
        output_info.append({
            "name": out.name,
            "shape": out.shape,
            "type": out.type
        })
    return {
        "model_name": model_name,
        "inputs": input_info,
        "outputs": output_info
    }

@app.get("/")
async def root():
    return {"Status": "The server is up and running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
