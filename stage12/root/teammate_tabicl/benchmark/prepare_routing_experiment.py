
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\mahidhar\Projects\DS Project\benchmark"
)

STAT_DIR = BASE_DIR / "representations" / "strategy_3_1"
TAB_DIR = BASE_DIR / "representations" / "strategy_3_2"

OUT_DIR = BASE_DIR / "representations" / "routing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRACTION = 0.20
K = 8

# Add known related dataset variants here.
# Every name must match dataset_names.json exactly.
#
# The automatic rules below are only a starting point.
# Review dataset_families.csv before treating the split
# as final.

MANUAL_FAMILIES = {
    "ada_family": [
        "ada",
        "ada_agnostic",
    ],
}

# ============================================================
# LOAD AND ALIGN REPRESENTATIONS
# ============================================================

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


stat_names = load_json(STAT_DIR / "dataset_names.json")
tab_names = load_json(TAB_DIR / "dataset_names.json")

if len(set(stat_names)) != len(stat_names):
    raise ValueError("Duplicate statistical dataset names")

if len(set(tab_names)) != len(tab_names):
    raise ValueError("Duplicate TabICL dataset names")

if set(stat_names) != set(tab_names):
    raise ValueError("Statistical and TabICL datasets differ")

tab_lookup = {
    name: i for i, name in enumerate(tab_names)
}

tab_order = [tab_lookup[name] for name in stat_names]

names = np.array(stat_names)

representations = {
    "stat_signed_log": np.load(
        STAT_DIR / "statistical_embeddings_raw.npy"
    ),
    "tab_mean_standard": np.load(
        TAB_DIR / "tabicl_mean_embeddings.npy"
    )[tab_order],
    "tab_mean_std_standard": np.load(
        TAB_DIR / "tabicl_mean_std_embeddings.npy"
    )[tab_order],
}

for key, X in representations.items():
    if X.ndim != 2 or X.shape[0] != len(names):
        raise ValueError(f"Invalid shape: {key} {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite embeddings: {key}")

    print(key, X.shape)

# ============================================================
# DATASET FAMILY IDENTIFICATION
# ============================================================

manual_lookup = {}

for family, members in MANUAL_FAMILIES.items():
    for member in members:
        if member in manual_lookup:
            raise ValueError(
                f"Duplicate manual family assignment: {member}"
            )
        manual_lookup[member] = family

unknown = set(manual_lookup) - set(names)

if unknown:
    raise ValueError(
        f"Unknown manual family members: {sorted(unknown)}"
    )


def get_family(name):
    if name in manual_lookup:
        return manual_lookup[name]

    lower = name.lower()

    # Conservative known-family rules.
    if lower.startswith("forex_"):
        return "FOREX_family"

    # Treat obvious dataset-name variants as a family.
    if lower.startswith("microwave"):
        return "microwave_family"

    # Otherwise keep each dataset independent.
    return f"individual::{name}"


families = np.array([get_family(name) for name in names])

family_df = pd.DataFrame({
    "dataset": names,
    "family": families,
})

family_df = family_df.sort_values(
    ["family", "dataset"]
)

family_df.to_csv(
    OUT_DIR / "dataset_families.csv",
    index=False,
)

print("\nTotal datasets:", len(names))
print("Total families:", len(set(families)))

print("\nLargest families:")
print(
    family_df.groupby("family")
    .size()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

# ============================================================
# GROUP-AWARE META-TRAIN / META-TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_FRACTION,
    random_state=SEED,
)

train_idx, test_idx = next(
    splitter.split(
        np.zeros(len(names)),
        groups=families,
    )
)

train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)

train_families = set(families[train_idx])
test_families = set(families[test_idx])

assert train_families.isdisjoint(test_families)
assert set(train_idx).isdisjoint(test_idx)
assert len(train_idx) + len(test_idx) == len(names)

if len(train_idx) <= K:
    raise ValueError(
        "Not enough meta-training datasets for K=8"
    )

split_labels = np.full(
    len(names),
    "meta_test",
    dtype=object,
)

split_labels[train_idx] = "meta_train"

manifest = pd.DataFrame({
    "dataset": names,
    "family": families,
    "split": split_labels,
})

print("\nMeta-training datasets:", len(train_idx))
print("Meta-test datasets:", len(test_idx))

# ============================================================
# TRAIN-ONLY PREPROCESSING AND CLUSTERING
# ============================================================

routing_summary = []

for config_name, X in representations.items():

    print(f"\nProcessing: {config_name}", flush=True)

    X = X.astype(np.float64, copy=True)

    # Match the exploratory comparison's transformation.
    if config_name == "stat_signed_log":
        X = np.sign(X) * np.log1p(np.abs(X))

    # Remove constant dimensions using meta-training
    # datasets only.
    variance = np.var(X[train_idx], axis=0)
    keep = variance > 1e-12

    if not np.any(keep):
        raise ValueError(
            f"No usable dimensions: {config_name}"
        )

    X = X[:, keep]

    # Fit scaling ONLY on meta-training datasets.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = KMeans(
        n_clusters=K,
        random_state=SEED,
        n_init=20,
    )

    train_clusters = model.fit_predict(X_train)

    # Assign held-out datasets to learned centroids.
    test_clusters = model.predict(X_test)

    column = f"{config_name}_cluster"

    manifest[column] = -1
    manifest.loc[train_idx, column] = train_clusters
    manifest.loc[test_idx, column] = test_clusters

    counts = np.bincount(
        train_clusters,
        minlength=K,
    )

    print("Training cluster sizes:", counts.tolist())

    # Save the fitted router.
    joblib.dump(
        {
            "configuration": config_name,
            "selected_dimensions": keep,
            "scaler": scaler,
            "kmeans": model,
            "signed_log": (
                config_name == "stat_signed_log"
            ),
            "seed": SEED,
        },
        OUT_DIR / f"router_{config_name}.joblib",
    )

    for cluster_id, count in enumerate(counts):
        routing_summary.append({
            "representation": config_name,
            "cluster": cluster_id,
            "n_meta_train": int(count),
            "n_meta_test": int(
                np.sum(test_clusters == cluster_id)
            ),
        })

# ============================================================
# SAVE RESULTS
# ============================================================

manifest.to_csv(
    OUT_DIR / "routing_manifest.csv",
    index=False,
)

summary_df = pd.DataFrame(routing_summary)

summary_df.to_csv(
    OUT_DIR / "routing_cluster_summary.csv",
    index=False,
)

with open(
    OUT_DIR / "experiment_config.json", "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "seed": SEED,
            "k": K,
            "test_fraction_requested": TEST_FRACTION,
            "n_datasets": len(names),
            "n_meta_train": len(train_idx),
            "n_meta_test": len(test_idx),
            "representations": list(representations),
            "note": (
                "Family groupings require manual review. "
                "Routers were fitted only on meta-training "
                "dataset embeddings."
            ),
        },
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("ROUTING EXPERIMENT SUMMARY")
print("=" * 70)

print("\nCluster membership counts:")
print(
    summary_df.to_string(index=False)
)

print("\nHeld-out datasets:")
print(
    manifest.loc[
        manifest["split"] == "meta_test",
        ["dataset", "family"]
    ].to_string(index=False)
)

print("\nSaved results to:", OUT_DIR)
print("Completed successfully.")
