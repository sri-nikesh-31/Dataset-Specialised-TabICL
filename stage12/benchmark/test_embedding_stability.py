
import json
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\mahidhar\Projects\DS Project\benchmark"
)
DATA_DIR = BASE_DIR / "dataset-250115" / "data"
STAT_DIR = BASE_DIR / "representations" / "strategy_3_1"
OUT_DIR = BASE_DIR / "representations" / "stability"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DATASETS = 12
N_REPEATS = 3
CONTEXT_ROWS = 128
SEED = 42

torch.set_num_threads(4)

with open(STAT_DIR / "dataset_names.json", encoding="utf-8") as f:
    all_names = json.load(f)

# Select datasets spread across the alphabetical list.
positions = np.linspace(
    0, len(all_names) - 1, N_DATASETS, dtype=int
)
selected_names = [all_names[i] for i in positions]

print("Selected datasets:")
for name in selected_names:
    print(" -", name)

# ============================================================
# LOAD AND SAMPLE
# ============================================================

def load_dataset(name):
    folder = DATA_DIR / name

    X = np.load(folder / "N_train.npy")
    y = np.load(folder / "y_train.npy", allow_pickle=True)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Invalid dataset shape")

    if len(X) < CONTEXT_ROWS + 1:
        raise ValueError("Dataset too small")

    if len(np.unique(y)) != 2:
        raise ValueError("Expected binary labels")

    return X, y


def make_sample(X, y, seed):
    rng = np.random.default_rng(seed)

    # Guarantee that both classes appear in context.
    classes = np.unique(y)
    chosen = []

    for cls in classes:
        candidates = np.flatnonzero(y == cls)
        chosen.append(int(rng.choice(candidates)))

    chosen = np.array(chosen, dtype=int)

    remaining = np.setdiff1d(
        np.arange(len(y)), chosen, assume_unique=False
    )

    extra = rng.choice(
        remaining,
        size=CONTEXT_ROWS - len(chosen),
        replace=False,
    )

    context_idx = np.concatenate([chosen, extra])
    rng.shuffle(context_idx)

    remaining = np.setdiff1d(
        np.arange(len(y)), context_idx, assume_unique=False
    )

    query_idx = rng.choice(remaining, size=1)

    # Fit imputation only on the sampled context.
    X_context = X[context_idx].copy()
    X_query = X[query_idx].copy()

    X_context[~np.isfinite(X_context)] = np.nan
    X_query[~np.isfinite(X_query)] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(X_context, axis=0)

    medians = np.where(
        np.isfinite(medians), medians, 0.0
    )

    def impute(A):
        missing = np.where(np.isnan(A))
        A[missing] = medians[missing[1]]
        return np.clip(A, -1e10, 1e10).astype(np.float32)

    return (
        impute(X_context),
        y[context_idx],
        impute(X_query),
    )

# ============================================================
# EXTRACT TABICL REPRESENTATIONS
# ============================================================

def extract_embeddings(X_context, y_context, X_query, seed):
    clf = TabICLClassifier(
        n_estimators=1,
        batch_size=1,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        random_state=seed,
        verbose=False,
    )

    clf.fit(X_context, y_context)
    clf.model_.eval()

    captured = []

    def hook(module, inputs, output):
        captured.append(output.detach().cpu().float())

    handle = clf.model_.row_interactor.register_forward_hook(hook)

    try:
        with torch.inference_mode():
            clf.predict_proba(X_query)
    finally:
        handle.remove()

    if len(captured) != 1:
        raise ValueError(
            f"Expected one hook call, got {len(captured)}"
        )

    H = captured[0]

    if tuple(H.shape) != (1, len(X_context) + 1, 512):
        raise ValueError(f"Unexpected shape: {tuple(H.shape)}")

    # Exclude the query row.
    H = H[0, :len(X_context)].numpy()

    mean = H.mean(axis=0)
    mean_std = np.concatenate(
        [mean, H.std(axis=0)]
    )

    if not (
        np.isfinite(mean).all()
        and np.isfinite(mean_std).all()
    ):
        raise ValueError("Non-finite embedding")

    del clf, H, captured
    gc.collect()

    return mean, mean_std

# ============================================================
# RUN EXPERIMENT
# ============================================================

records = []
failures = []

for i, name in enumerate(selected_names, 1):
    print(f"\n[{i}/{len(selected_names)}] {name}", flush=True)

    try:
        X, y = load_dataset(name)

        for repeat in range(N_REPEATS):
            seed = SEED + repeat * 1009

            Xc, yc, Xq = make_sample(X, y, seed)

            z_mean, z_mean_std = extract_embeddings(
                Xc, yc, Xq, seed
            )

            records.append({
                "dataset": name,
                "repeat": repeat,
                "mean": z_mean,
                "mean_std": z_mean_std,
            })

            print(
                f"  Repeat {repeat + 1}/{N_REPEATS}: SUCCESS",
                flush=True,
            )

    except Exception as exc:
        print("  FAILED:", str(exc), flush=True)
        failures.append({
            "dataset": name,
            "error": str(exc),
        })

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

with open(OUT_DIR / "failures.json", "w") as f:
    json.dump(failures, f, indent=2)

if not records:
    raise RuntimeError("No embeddings were generated")

metadata = pd.DataFrame([
    {
        "dataset": r["dataset"],
        "repeat": r["repeat"],
    }
    for r in records
])

metadata.to_csv(
    OUT_DIR / "sample_metadata.csv",
    index=False,
)

# ============================================================
# ANALYZE WITHIN / BETWEEN DATASET SIMILARITY
# ============================================================

summary_rows = []

for representation in ["mean", "mean_std"]:
    Z = np.stack([r[representation] for r in records])
    np.save(
        OUT_DIR / f"{representation}_resampled.npy",
        Z,
    )

    similarity = cosine_similarity(Z)
    np.fill_diagonal(similarity, np.nan)

    labels = metadata["dataset"].to_numpy()

    same = labels[:, None] == labels[None, :]
    different = ~same

    within = similarity[same]
    between = similarity[different]

    within = within[np.isfinite(within)]
    between = between[np.isfinite(between)]

    # Leave-one-sample-out nearest-neighbour retrieval.
    correct = 0
    eligible = 0

    for row in range(len(Z)):
        other_same = np.sum(labels == labels[row]) - 1

        if other_same == 0:
            continue

        scores = similarity[row].copy()
        scores[np.isnan(scores)] = -np.inf

        neighbor = int(np.argmax(scores))

        eligible += 1
        correct += int(labels[neighbor] == labels[row])

    accuracy = correct / eligible if eligible else np.nan

    summary_rows.append({
        "representation": representation,
        "n_samples": len(Z),
        "within_mean": float(np.mean(within)),
        "within_std": float(np.std(within)),
        "between_mean": float(np.mean(between)),
        "between_std": float(np.std(between)),
        "similarity_gap": float(
            np.mean(within) - np.mean(between)
        ),
        "nearest_neighbor_accuracy": float(accuracy),
    })

    # Histogram: similarity distributions.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        within,
        bins=25,
        alpha=0.6,
        density=True,
        label="Same dataset",
    )
    ax.hist(
        between,
        bins=25,
        alpha=0.6,
        density=True,
        label="Different datasets",
    )

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Embedding stability: {representation}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / f"similarity_{representation}.png",
        dpi=180,
    )
    plt.close(fig)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT_DIR / "stability_summary.csv",
    index=False,
)

print("\n" + "=" * 70)
print("EMBEDDING STABILITY RESULTS")
print("=" * 70)
print(summary.round(4).to_string(index=False))
print("\nFailures:", len(failures))
print("Output directory:", OUT_DIR)
print("Completed.")
