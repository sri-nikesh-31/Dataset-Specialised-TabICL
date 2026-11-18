
import json
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from tabicl import TabICLClassifier

BASE_DIR = Path(
    r"C:\mahidhar\Projects\DS Project\benchmark"
)

OUTPUT_DIR = BASE_DIR / "representations" / "strategy_3_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X, y = make_classification(
    n_samples=160,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42,
)

X = X.astype(np.float32)

clf = TabICLClassifier(
    n_estimators=1,
    batch_size=1,
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    random_state=42,
    verbose=False,
)

print("TabICL:", importlib.metadata.version("tabicl"))
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

clf.fit(X[:120], y[:120])

model = clf.model_
model.eval()

captured = []
records = []

def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}

    return {"type": type(value).__name__}

def capture_hook(module, inputs, output):
    record = {
        "input": describe(inputs),
        "output": describe(output),
    }
    records.append(record)

    def collect(value):
        if isinstance(value, torch.Tensor):
            captured.append(value.detach().cpu())
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(output)

handle = model.row_interactor.register_forward_hook(
    capture_hook
)

try:
    with torch.inference_mode():
        probabilities = clf.predict_proba(X[120:125])
finally:
    handle.remove()

print("\nPrediction shape:", probabilities.shape)
print("Hook calls:", len(records))

for i, record in enumerate(records):
    print(f"\nHook call {i + 1}")
    print(json.dumps(record, indent=2))

print("\nCaptured tensors:", len(captured))

for i, tensor in enumerate(captured):
    print(
        f"Tensor {i}:",
        "shape =", tuple(tensor.shape),
        "finite =", bool(torch.isfinite(tensor).all())
    )

report = {
    "tabicl_version": importlib.metadata.version("tabicl"),
    "prediction_shape": list(probabilities.shape),
    "hook_calls": records,
    "captured_tensor_shapes": [
        list(t.shape) for t in captured
    ],
}

report_path = OUTPUT_DIR / "embedding_probe.json"

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved:", report_path)
