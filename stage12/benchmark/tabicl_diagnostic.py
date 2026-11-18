
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

print("=" * 70)
print("TABICL REPRESENTATION EXTRACTION DIAGNOSTIC")
print("=" * 70)

print("TabICL version:",
      importlib.metadata.version("tabicl"))
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Small synthetic dataset for checking the model.
X, y = make_classification(
    n_samples=400,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42,
)

clf = TabICLClassifier(
    n_estimators=1,
    batch_size=1,
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    random_state=42,
    verbose=True,
)

print("\nInitializing pretrained model...")
clf.fit(X[:300], y[:300])

print("\nRunning inference...")
pred = clf.predict_proba(X[300:310])

print("Prediction shape:", pred.shape)
print("All predictions finite:",
      bool(np.isfinite(pred).all()))

print("\nClassifier attributes:")
for name in sorted(vars(clf)):
    value = getattr(clf, name)

    if isinstance(value, torch.nn.Module):
        print(
            name,
            "->",
            type(value).__name__,
            "parameters:",
            sum(p.numel() for p in value.parameters())
        )
    else:
        print(name, "->", type(value).__name__)

# Locate PyTorch modules, including nested model objects.
found = []
visited = set()

def inspect_object(obj, path, depth=0):
    if depth > 3 or id(obj) in visited:
        return

    visited.add(id(obj))

    if isinstance(obj, torch.nn.Module):
        found.append((path, obj))
        return

    if not hasattr(obj, "__dict__"):
        return

    for name, value in vars(obj).items():
        if name.startswith("__"):
            continue

        if isinstance(value, torch.nn.Module):
            found.append((f"{path}.{name}", value))
        elif (
            depth < 3
            and not isinstance(
                value,
                (str, int, float, bool, np.ndarray,
                 list, tuple, dict, Path)
            )
        ):
            inspect_object(
                value, f"{path}.{name}", depth + 1
            )

inspect_object(clf, "clf")

print("\nLocated PyTorch models:", len(found))

for path, model in found:
    print("\nMODEL PATH:", path)
    print("MODEL TYPE:", type(model).__name__)

    print("\nTop-level child modules:")
    for name, module in model.named_children():
        print(" ", name, "->", type(module).__name__)

    print("\nModules containing encoder/transformer:")
    for name, module in model.named_modules():
        if any(
            term in name.lower()
            for term in [
                "encoder", "transformer",
                "column", "row", "feature"
            ]
        ):
            print(name, "->", type(module).__name__)

print("\nDiagnostic completed.")
