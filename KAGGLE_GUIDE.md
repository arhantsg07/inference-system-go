# Running on Kaggle (Dual T4 GPUs)

This guide covers deploying the inference system on Kaggle with dual NVIDIA T4 GPUs, leveraging Tensor Cores for maximum throughput.

## T4 Tensor Cores

The NVIDIA T4 GPU has **320 Tensor Cores** (Turing architecture) that accelerate:
- **FP16** (half-precision): ~2x throughput vs FP32
- **INT8** (integer): ~4x throughput vs FP32
- **INT4** (experimental): ~8x throughput vs FP32

Tensor Cores are automatically engaged when using CUDA execution provider with supported precision.

## Quick Start on Kaggle

### 1. Create a Kaggle Notebook

- GPU type: **T4 x2** (enable under Settings → Accelerator)
- Framework: **Custom (no predefined image)**
- Internet: **Enabled** (for pip installs)

### 2. Setup Script

Run this in a Kaggle notebook cell:

```python
!pip install fastapi uvicorn onnxruntime-gpu opencv-python-headless ultralytics numpy
```

### 3. Upload the project

Add this repo as a Kaggle Dataset or upload via the notebook:

```python
import os, zipfile
# If uploaded as dataset
!unzip /kaggle/input/your-dataset/inference-system-go.zip -d /kaggle/working/
os.chdir("/kaggle/working/inference-system-go")
```

### 4. Export YOLOv8n and Quantize

```python
from ultralytics import YOLO
from pathlib import Path

MODELS_DIR = Path("/kaggle/working/inference-system-go/models")

# Export FP32
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)

# Move to models dir
import shutil
shutil.move("yolov8n.onnx", MODELS_DIR / "yolov8n.onnx")

# Quantize to FP16
import onnx
from onnxconverter_common import float16
model_fp32 = onnx.load(str(MODELS_DIR / "yolov8n.onnx"))
model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
while model_fp16.graph.value_info:
    model_fp16.graph.value_info.pop()
model_fp16 = onnx.shape_inference.infer_shapes(model_fp16, strict_mode=False)
onnx.save(model_fp16, str(MODELS_DIR / "yolov8n_fp16.onnx"))

# Quantize to INT8 (dynamic)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    str(MODELS_DIR / "yolov8n.onnx"),
    str(MODELS_DIR / "yolov8n_int8.onnx"),
    weight_type=QuantType.QInt8,
)
```

### 5. Run Python Model Server (background)

```python
import subprocess, time
server_proc = subprocess.Popen(
    ["python3", "services/model_server/main.py"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(3)
print("Server PID:", server_proc.pid)
```

## Benchmarking on Dual T4

### GPU Benchmark Script

```python
import cv2
import numpy as np
import onnxruntime as ort
import time
import statistics

MODELS_DIR = Path("/kaggle/working/inference-system-go/models")
IMAGE_PATH = MODELS_DIR / "sample_test.jpg"
WARMUP = 50
ITERATIONS = 500

def benchmark(model_name, provider):
    print(f"\n--- {model_name} ({provider}) ---")
    path = MODELS_DIR / f"{model_name}.onnx"

    # Session options for Tensor Core utilization
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = [provider, "CPUExecutionProvider"]
    provider_opts = None
    if provider == "CUDAExecutionProvider":
        provider_opts = [{
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }]

    sess = ort.InferenceSession(str(path), sess_options=sess_opts,
                                providers=providers, provider_options=provider_opts)

    # Preprocess once
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(WARMUP):
        sess.run(None, {input_name: img})

    # Benchmark
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        sess.run(None, {input_name: img})
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    s = sorted(times)
    n = len(s)
    print(f"  Mean:     {statistics.mean(times):.2f} ms")
    print(f"  Median:   {s[n//2]:.2f} ms")
    print(f"  P95:      {s[int(n*0.95)]:.2f} ms")
    print(f"  P99:      {s[int(n*0.99)]:.2f} ms")
    print(f"  Throughput: {1000/statistics.mean(times):.1f} req/s")
    return statistics.mean(times)

models = ["yolov8n", "yolov8n_fp16", "yolov8n_int8"]
results = {}
for m in models:
    results[m] = benchmark(m, "CUDAExecutionProvider")

print("\n=== Summary ===")
base = results["yolov8n"]
for m, t in results.items():
    print(f"  {m:20s}: {t:.2f} ms  ({base/t:.2f}x vs FP32)")
```

### Expected Results on Dual T4

| Model | Precision | Single T4 | Dual T4* | Tensor Cores |
|-------|-----------|-----------|----------|--------------|
| `yolov8n.onnx` | FP32 | ~3 ms | ~3 ms | No (FP32) |
| `yolov8n_fp16.onnx` | FP16 | **~1.5 ms** | ~1.5 ms | **Yes** |
| `yolov8n_int8.onnx` | INT8 | **~0.8 ms** | ~0.8 ms | **Yes** |

*Note: Dual GPU requires manual load balancing (e.g., shard requests across both GPUs). Single GPU inference doesn't automatically scale across both.

## Tensor Core Configuration

Tensor Cores activate automatically when using `CUDAExecutionProvider` with FP16 or INT8 inputs. Key requirements:

### For FP16 (Tensor Cores active):
```python
sess = ort.InferenceSession(
    "yolov8n_fp16.onnx",
    providers=["CUDAExecutionProvider"]
)
```

### For INT8 (Tensor Cores active):
```python
sess = ort.InferenceSession(
    "yolov8n_int8.onnx",
    providers=["CUDAExecutionProvider"]
)
```

### Session options to maximize Tensor Core utilization:
```python
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_opts.enable_cpu_mem_arena = False
```

## Multi-GPU Setup (Dual T4)

### Option A: Two independent servers (port per GPU)

```python
import subprocess, os

# Server for GPU 0
env0 = os.environ.copy()
env0["CUDA_VISIBLE_DEVICES"] = "0"
srv0 = subprocess.Popen(
    ["python3", "services/model_server/main.py", "--port", "8080"],
    env=env0
)

# Server for GPU 1
env1 = os.environ.copy()
env1["CUDA_VISIBLE_DEVICES"] = "1"
srv1 = subprocess.Popen(
    ["python3", "services/model_server/main.py", "--port", "8081"],
    env=env1
)
```

### Option B: Round-robin load balancer (Python)

```python
import itertools, json, urllib.request, base64

gpu_ports = itertools.cycle(["8080", "8081"])

def predict(image_path):
    port = next(gpu_ports)
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({"model_name": "yolov8n", "input": img_b64}).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/predict",
        data=payload, headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())

# Batch inference across dual GPUs
for i in range(100):
    result = predict("models/sample_test.jpg")
    print(f"Request {i}: {result['inference_time_ms']:.2f}ms")
```

## Verifying Tensor Core Usage

```bash
# Check GPU utilization and Tensor Core activity
nvidia-smi -l 1
```

Look for high `Volatile GPU-Util` during FP16/INT8 inference. You can also profile with:

```bash
nsys profile -o trace python3 benchmark_gpu.py
```

## Important Notes

- **Kaggle session limits**: 12 hours max, 30 hours/week. Download models at session start
- **Dataset persistence**: Place models in `/kaggle/working/` for notebook-session lifetime
- **Memory**: T4 has 16 GB VRAM per GPU. YOLOv8n uses <1 GB
- **First inference slower**: CUDA kernel compilation adds ~1-2s on first run — always include warmup
