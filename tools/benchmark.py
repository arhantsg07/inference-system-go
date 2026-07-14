"""
Benchmark script for YOLOv8n ONNX inference.
Measures latency (mean, median, p95, p99) and throughput across multiple runs.
"""
import argparse
import base64
import json
import time
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def run_inference(session: ort.InferenceSession, input_tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})


def benchmark_model(model_path: Path, image_path: Path, label: str, provider: str = "CPUExecutionProvider"):
    print(f"\n{'='*60}")
    print(f"Benchmark: {label} ({provider})")
    print(f"Model: {model_path.name} ({model_path.stat().st_size / 1e6:.2f} MB)")
    print(f"{'='*60}")

    providers = [provider, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_tensor = preprocess_image(str(image_path))

    # GPU warmup (first inference triggers CUDA kernel compilation)
    _ = run_inference(session, input_tensor)

    # Warmup iterations
    for _ in range(WARMUP_ITERATIONS):
        _ = run_inference(session, input_tensor)

    # Benchmark pipeline (preprocessing + inference)
    pipeline_times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        tensor = preprocess_image(str(image_path))
        _ = run_inference(session, tensor)
        elapsed = (time.perf_counter() - start) * 1000
        pipeline_times.append(elapsed)

    # Benchmark pure inference (already preprocessed)
    inference_times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        _ = run_inference(session, input_tensor)
        elapsed = (time.perf_counter() - start) * 1000
        inference_times.append(elapsed)

    def stats(times):
        sorted_t = sorted(times)
        n = len(sorted_t)
        return {
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(sorted_t[n // 2], 2),
            "p95_ms": round(sorted_t[int(n * 0.95)], 2),
            "p99_ms": round(sorted_t[int(n * 0.99)], 2),
            "min_ms": round(sorted_t[0], 2),
            "max_ms": round(sorted_t[-1], 2),
            "std_ms": round(statistics.stdev(times), 2),
            "throughput_rps": round(1000 / statistics.mean(times), 1),
        }

    pipe_stats = stats(pipeline_times)
    infer_stats = stats(inference_times)

    print(f"\n{'Metric':<25} {'Pipeline':<15} {'Inference Only':<15}")
    print(f"{'-'*55}")
    for key in ["mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "std_ms"]:
        label_key = key.replace("_ms", " (ms)").replace("_", " ").title()
        print(f"{label_key:<25} {pipe_stats[key]:<15} {infer_stats[key]:<15}")
    print(f"\n{'Throughput (req/s)':<25} {pipe_stats['throughput_rps']:<15} {infer_stats['throughput_rps']:<15}")

    return pipe_stats, infer_stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 ONNX models")
    parser.add_argument("--models", nargs="+", default=["yolov8n"],
                        help="Model names to benchmark (without .onnx)")
    parser.add_argument("--image", default=str(MODELS_DIR / "sample_test.jpg"),
                        help="Path to test image")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS,
                        help=f"Number of benchmark iterations (default: {BENCHMARK_ITERATIONS})")
    parser.add_argument("--provider", default="CPUExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                        help="ONNX Runtime execution provider")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    results = {}
    for model_name in args.models:
        model_path = MODELS_DIR / f"{model_name}.onnx"
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            continue

        pipe_stats, infer_stats = benchmark_model(
            model_path, image_path, model_name, provider=args.provider
        )
        results[model_name] = {"pipeline": pipe_stats, "inference_only": infer_stats}

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        base_model = args.models[0]
        base_pipe = results[base_model]["pipeline"]["mean_ms"]
        base_infer = results[base_model]["inference_only"]["mean_ms"]

        print(f"\n{'Model':<20} {'Pipeline (ms)':<20} {'Inference (ms)':<20} {'Speedup %':<15}")
        print(f"{'-'*75}")
        for model_name, stats in results.items():
            p = stats["pipeline"]["mean_ms"]
            i = stats["inference_only"]["mean_ms"]
            speedup_p = ((base_pipe - p) / base_pipe) * 100
            speedup_i = ((base_infer - i) / base_infer) * 100
            print(f"{model_name:<20} {p:<20} {i:<20} {speedup_i:<15.1f}")


if __name__ == "__main__":
    main()
