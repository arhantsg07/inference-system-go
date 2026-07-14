"""
Quantize YOLOv8n ONNX model to FP16 and INT8.
"""
from pathlib import Path
import onnx
from onnxconverter_common import float16

from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "yolov8n.onnx"


def to_fp16():
    print("Converting to FP16...")
    model = onnx.load(str(MODEL_PATH))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    # Remove duplicate/conflicting value_info entries
    while model_fp16.graph.value_info:
        model_fp16.graph.value_info.pop()
    model_fp16 = onnx.shape_inference.infer_shapes(model_fp16, strict_mode=False)

    out_path = MODELS_DIR / "yolov8n_fp16.onnx"
    onnx.save(model_fp16, str(out_path))
    orig_size = MODEL_PATH.stat().st_size / 1e6
    new_size = out_path.stat().st_size / 1e6
    print(f"  FP32: {orig_size:.2f} MB -> FP16: {new_size:.2f} MB ({new_size/orig_size*100:.1f}%)")


def to_int8():
    print("Converting to INT8 (dynamic quantization)...")
    out_path = MODELS_DIR / "yolov8n_int8.onnx"
    quantize_dynamic(
        str(MODEL_PATH),
        str(out_path),
        weight_type=QuantType.QInt8,
    )
    orig_size = MODEL_PATH.stat().st_size / 1e6
    new_size = out_path.stat().st_size / 1e6
    print(f"  FP32: {orig_size:.2f} MB -> INT8: {new_size:.2f} MB ({new_size/orig_size*100:.1f}%)")


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        exit(1)

    to_fp16()
    to_int8()

    print("\nAll quantized models:")
    for f in sorted(MODELS_DIR.glob("*.onnx")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")
