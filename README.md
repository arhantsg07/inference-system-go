# ML Inference System

A production-grade **gRPC-based ML inference system** featuring YOLOv8 object detection with ONNX Runtime, FP16/INT8 quantization, GPU acceleration (CUDA), Prometheus metrics, and comprehensive benchmarking.

## Architecture

```
Client (Go/gRPC) → Go gRPC Server (:50051) → Python FastAPI Server (:8080) → ONNX Runtime → YOLOv8 Model
                                                                                         ↓
                                                                               Prometheus Metrics (:9090)
```

## Features

- **gRPC API** for inference requests with context deadlines
- **YOLOv8n** object detection via ONNX Runtime (CPU / CUDA GPU)
- **Quantization support**: FP16 and INT8 model variants
- **Model size**: FP32 (12.85 MB) → FP16 (6.45 MB) → INT8 (3.50 MB)
- **Prometheus metrics**: request count, latency histograms
- **Benchmark suite**: mean, median, p95, p99 latency + throughput

## Quick Start

### Prerequisites

- Go 1.21+
- Python 3.10+

### Setup

```bash
# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r services/model_server/requirements.txt

# Go dependencies
go mod tidy
```

### Run the system

```bash
# Terminal 1: Start Python model server
source .venv/bin/activate
python3 services/model_server/main.py

# Terminal 2: Start Go gRPC server
go run cmd/server/main.go

# Terminal 3: Run inference client
go run cmd/client/main.go models/sample_test.jpg
```

### Models

| Model | Format | Size | vs FP32 |
|---|---|---|---|
| `yolov8n.onnx` | FP32 | 12.85 MB | — |
| `yolov8n_fp16.onnx` | FP16 | 6.45 MB | 50.2% |
| `yolov8n_int8.onnx` | INT8 | 3.50 MB | 27.2% |

## Benchmark Results

**Hardware:** CPU: 11th Gen Intel Core i5-11400H, GPU: NVIDIA GeForce GTX 1650 (4GB, no Tensor Cores)

### CPU (ONNX Runtime)

| Metric | FP32 | FP16 | INT8 |
|---|---|---|---|
| **Inference Mean** | 49.21 ms | 52.25 ms | 161.29 ms |
| **Pipeline Mean** | 51.75 ms | 56.27 ms | 153.04 ms |
| **Throughput** | 20.3 req/s | 19.1 req/s | 6.2 req/s |

### GPU / CUDA (ONNX Runtime)

| Metric | FP32 | FP16 | INT8 |
|---|---|---|---|
| **Inference Mean** | **11.99 ms** | 30.82 ms | 235.10 ms |
| **Throughput** | **83.4 req/s** | 32.4 req/s | 4.3 req/s |
| **vs CPU FP32** | **4.1x faster** | — | — |

### Key findings

- **GPU inference (FP32):** 11.99 ms — **4.1x faster** than CPU (49.21 ms), the optimal configuration for this GPU
- **FP16 quantization:** 50% model size reduction. On GPUs without Tensor Cores (GTX 1650), FP16 is slower than FP32 due to Cast node overhead from Resize ops. On Tensor Core GPUs (RTX series), FP16 provides significant speedup
- **INT8 dynamic quantization:** 73% model compression for memory-constrained deployment. On GPUs, dynamic INT8 adds dequantize overhead — static INT8 quantization (via TensorRT) is recommended for GPU speedup
- **CPU:** FP32 provides best latency (49.21 ms); FP16 offers size savings with negligible latency impact
- All quantized models maintain detection accuracy (max confidence within ±3% of FP32)

## Benchmarking

```bash
# CPU benchmarks
source .venv/bin/activate
python3 tools/benchmark.py --models yolov8n yolov8n_fp16 yolov8n_int8

# GPU benchmarks (requires CUDA)
export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/nvidia/cudnn/lib:.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
python3 tools/benchmark.py --models yolov8n --provider CUDAExecutionProvider
```

## Model Export & Quantization

```bash
# Export YOLOv8 from PyTorch to ONNX
source .venv/bin/activate
python3 tools/export_model.py yolov8n

# Quantize to FP16 and INT8
python3 tools/quantize.py
```

## API

### gRPC (port 50051)

```protobuf
rpc Predict(PredictRequest) returns (PredictResponse)
```

### REST (port 8080, Python server)

```
POST /predict          Run inference (JSON with base64 image)
GET  /models           List available models
GET  /model_info/{id}  Get model input/output details
```

### Metrics (port 9090)

```
GET /metrics    Prometheus metrics
GET /health     Health check
```