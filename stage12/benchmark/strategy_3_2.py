
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(
    r"C:\mahidhar\Projects\DS Project\benchmark"
)
DATA_DIR = BASE_DIR / "dataset-250115" / "data"
OUTPUT_DIR = BASE_DIR / "representations" / "strategy_3_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAT_DIR = BASE_DIR / "representations" / "strategy_3_1"

SEED = 42
CONTEXT_ROWS = 256

with open(STAT_DIR / "dataset_names.json") as f:
    dataset_names = json.load(f)

mean_embeddings = []
mean_std_embeddings = []
successful_names = []
failures = []

print("Device: CPU")
print("Datasets:", len(dataset_names))

for idx, name in enumerate(dataset_names, 1):
    print(f"\n[{idx}/{len(dataset_names)}] {name}", flush=True)

    try:
        folder = DATA_DIR / name

        X = np.load(folder / "N_train.npy")
        y = np.load(
            folder / "y_train.npy",
            allow_pickle=True
        )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        if len(X) <= CONTEXT_ROWS:
            raise ValueError("Not enough observations")

        # Replace non-finite values using training-data medians.
        X[~np.isfinite(X)] = np.nan

        medians = np.nanmedian(X, axis=0)
        medians = np.where(
            np.isfinite(medians), medians, 0.0
        )

        missing = np.where(np.isnan(X))
        X[missing] = medians[missing[1]]

        X = np.clip(X, -1e10, 1e10).astype(np.float32)

        indices = np.arange(len(y))

        context_idx, remaining_idx = train_test_split(
            indices,
            train_size=CONTEXT_ROWS,
            stratify=y,
            random_state=SEED,
        )

        query_idx = remaining_idx[:1]

        X_context = X[context_idx]
        y_context = y[context_idx]
        X_query = X[query_idx]

        clf = TabICLClassifier(
            n_estimators=1,
            batch_size=1,
            checkpoint_version=(
                "tabicl-classifier-v2-20260212.ckpt"
            ),
            random_state=SEED,
            verbose=False,
        )

        clf.fit(X_context, y_context)
        clf.model_.eval()

        captured = []

        def hook(module, inputs, output):
            captured.append(
                output.detach().cpu().float().clone()
            )

        handle = clf.model_.row_interactor.register_forward_hook(
            hook
        )

        try:
            with torch.inference_mode():
                clf.predict_proba(X_query)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise ValueError(
                f"Expected 1 hook call, got {len(captured)}"
            )

        H = captured[0]

        if H.ndim != 3 or H.shape[0] != 1:
            raise ValueError(
                f"Unexpected embedding shape: {H.shape}"
            )

        if H.shape[1] != CONTEXT_ROWS + len(X_query):
            raise ValueError(
                f"Unexpected row count: {H.shape}"
            )

        # Exclude query row: retain context representations.
        H = H[0, :CONTEXT_ROWS, :].numpy()

        if H.shape[1] != 512:
            raise ValueError(
                f"Expected 512 dimensions, got {H.shape[1]}"
            )

        z_mean = H.mean(axis=0)
        z_std = H.std(axis=0)

        z_mean_std = np.concatenate([
            z_mean, z_std
        ])

        if not (
            np.isfinite(z_mean).all()
            and np.isfinite(z_mean_std).all()
        ):
            raise ValueError("Non-finite embeddings")

        mean_embeddings.append(z_mean)
        mean_std_embeddings.append(z_mean_std)
        successful_names.append(name)

        print(
            "SUCCESS:",
            "mean =", z_mean.shape,
            "mean+std =", z_mean_std.shape,
            flush=True,
        )

        # Save progress after each dataset.
        np.save(
            OUTPUT_DIR / "tabicl_mean_embeddings.npy",
            np.asarray(mean_embeddings)
        )
        np.save(
            OUTPUT_DIR / "tabicl_mean_std_embeddings.npy",
            np.asarray(mean_std_embeddings)
        )

        with open(
            OUTPUT_DIR / "dataset_names.json", "w"
        ) as f:
            json.dump(successful_names, f, indent=2)

    except Exception as e:
        print("FAILED:", str(e), flush=True)
        failures.append({
            "dataset": name,
            "error": str(e)
        })

    with open(OUTPUT_DIR / "failures.json", "w") as f:
        json.dump(failures, f, indent=2)

print("\n" + "=" * 60)
print("STRATEGY 3.2 RESULTS")
print("=" * 60)

print("Successful:", len(successful_names))
print("Failed:", len(failures))

if successful_names:
    print(
        "Mean embeddings:",
        np.asarray(mean_embeddings).shape
    )
    print(
        "Mean+Std embeddings:",
        np.asarray(mean_std_embeddings).shape
    )

print("Output:", OUTPUT_DIR)
