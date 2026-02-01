import os
import torch
from torch import nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "sample.onnx")

# A tiny sample model 
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SampleModel().to(device)
    model.eval()

    sample = torch.randn(1, 3, dtype=torch.float32, device=device)

    torch.onnx.export(
        model, 
        sample, 
        OUT_PATH,
        export_params=True, opset_version=12,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"Exported ONNX model to: {OUT_PATH}")

    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": np.random.randn(1, 3).astype(np.float32)})
        print("ONNX inference output:", out[0])
    except Exception:
        pass

