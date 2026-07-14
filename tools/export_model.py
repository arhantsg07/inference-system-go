from ultralytics import YOLO
from pathlib import Path
import sys

OUT_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"

if __name__ == "__main__":
    model = YOLO(f"{MODEL_NAME}.pt")
    out_path = OUT_DIR / f"{MODEL_NAME}.onnx"
    model.export(format="onnx", imgsz=640, simplify=True)
    print(f"Model exported to: {out_path}")
