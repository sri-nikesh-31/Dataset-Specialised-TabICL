
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    normalize,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\mahidhar\Projects\DS Project\benchmark"
)

STAT_DIR = BASE_DIR / "representations" / "strategy_3_1"
TAB_DIR = BASE_DIR / "representations" / "strategy_3_2"

OUT_DIR = BASE_DIR / "representations" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAIN_K = 8
K_VALUES = range(2, 13)
STABILITY_RUNS = 30

# ============================================================
# LOAD DATA
# ============================================================

def load_names(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


stat_names = load_names(STAT_DIR / "dataset_names.json")
tab_names = load_names(TAB_DIR / "dataset_names.json")

if len(stat_names) != len(set(stat_names)):
    raise ValueError("Duplicate statistical dataset names")

if len(tab_names) != len(set(tab_names)):
    raise ValueError("Duplicate TabICL dataset names")

if set(stat_names) != set(tab_names):
    missing_stat = sorted(set(tab_names) - set(stat_names))
    missing_tab = sorted(set(stat_names) - set(tab_names))
    raise ValueError(
        f"Dataset mismatch:\n"
        f"Missing statistical: {missing_stat}\n"
        f"Missing TabICL: {missing_tab}"
    )

# Reorder TabICL rows to match the statistical baseline.
tab_index = {name: i for i, name in enumerate(tab_names)}
order = [tab_index[name] for name in stat_names]

names = stat_names

raw = {
    "statistical": np.load(
        STAT_DIR / "statistical_embeddings_raw.npy"
    ),
    "tabicl_mean": np.load(
        TAB_DIR / "tabicl_mean_embeddings.npy"
    )[order],
    "tabicl_mean_std": np.load(
        TAB_DIR / "tabicl_mean_std_embeddings.npy"
    )[order],
}

for name, X in raw.items():
    if X.ndim != 2 or X.shape[0] != len(names):
        raise ValueError(f"Invalid shape for {name}: {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in {name}")
    print(name, X.shape)

# ============================================================
# PREPROCESSING
# ============================================================

def prepare(X, method):
    X = np.asarray(X, dtype=np.float64)

    # Remove dimensions with zero variance.
    keep = np.var(X, axis=0) > 1e-12
    X = X[:, keep]

    if method == "standard":
        X = StandardScaler().fit_transform(X)

    elif method == "robust":
        X = RobustScaler().fit_transform(X)

    elif method == "unit":
        X = normalize(X, norm="l2")

    elif method == "signed_log_standard":
        X = np.sign(X) * np.log1p(np.abs(X))
        X = StandardScaler().fit_transform(X)

    elif method != "raw":
        raise ValueError(method)

    return X


configurations = {
    "stat_standard": ("statistical", "standard"),
    "stat_robust": ("statistical", "robust"),
    "stat_signed_log": (
        "statistical", "signed_log_standard"
    ),
    "tab_mean_raw": ("tabicl_mean", "raw"),
    "tab_mean_standard": ("tabicl_mean", "standard"),
    "tab_mean_unit": ("tabicl_mean", "unit"),
    "tab_mean_std_raw": ("tabicl_mean_std", "raw"),
    "tab_mean_std_standard": (
        "tabicl_mean_std", "standard"
    ),
    "tab_mean_std_unit": ("tabicl_mean_std", "unit"),
}

processed = {
    key: prepare(raw[rep], method)
    for key, (rep, method) in configurations.items()
}

# ============================================================
# CLUSTERING
# ============================================================

def cluster(X, k, seed=SEED):
    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=20,
    )
    return model.fit_predict(X)


records = []
assignments = pd.DataFrame({"dataset": names})

for config_name, X in processed.items():
    print("\nProcessing:", config_name, flush=True)

    for k in K_VALUES:
        labels = cluster(X, k)

        counts = np.bincount(labels, minlength=k)

        records.append({
            "representation": config_name,
            "k": k,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(
                X, labels
            ),
            "smallest_cluster": int(counts.min()),
            "largest_cluster": int(counts.max()),
            "singleton_clusters": int(
                np.sum(counts == 1)
            ),
        })

        if k == MAIN_K:
            assignments[config_name] = labels

metrics = pd.DataFrame(records)

metrics.to_csv(
    OUT_DIR / "clustering_metrics.csv",
    index=False,
)

assignments.to_csv(
    OUT_DIR / "cluster_assignments_k8.csv",
    index=False,
)

# ============================================================
# STABILITY TEST 1: DIFFERENT K-MEANS INITIALIZATIONS
# ============================================================

# ARI compares partitions while ignoring cluster label IDs.

seed_records = []

for config_name, X in processed.items():
    baseline = cluster(X, MAIN_K, seed=SEED)

    scores = []

    for seed in range(STABILITY_RUNS):
        candidate = cluster(
            X, MAIN_K, seed=1000 + seed
        )

        scores.append(
            adjusted_rand_score(baseline, candidate)
        )

    seed_records.append({
        "representation": config_name,
        "seed_ari_mean": float(np.mean(scores)),
        "seed_ari_std": float(np.std(scores)),
    })

# ============================================================
# STABILITY TEST 2: DATASET SUBSAMPLING
# ============================================================

# Recluster 80% of the datasets and compare assignments
# against the original partition on those same datasets.
# This is a preliminary sensitivity test, not a replacement
# for row-resampling stability of the embeddings.

rng = np.random.default_rng(SEED)
subsample_records = []

for config_name, X in processed.items():
    baseline = cluster(X, MAIN_K)

    scores = []

    for run in range(STABILITY_RUNS):
        subset = np.sort(
            rng.choice(
                len(X),
                size=int(0.8 * len(X)),
                replace=False,
            )
        )

        candidate = cluster(
            X[subset],
            MAIN_K,
            seed=2000 + run,
        )

        scores.append(
            adjusted_rand_score(
                baseline[subset],
                candidate,
            )
        )

    subsample_records.append({
        "representation": config_name,
        "subsample_ari_mean": float(np.mean(scores)),
        "subsample_ari_std": float(np.std(scores)),
    })

stability = pd.DataFrame(seed_records).merge(
    pd.DataFrame(subsample_records),
    on="representation",
)

stability.to_csv(
    OUT_DIR / "clustering_stability.csv",
    index=False,
)

# ============================================================
# NEAREST NEIGHBOURS
# ============================================================

neighbor_records = []

for config_name, X in processed.items():

    # Cosine distance is used consistently here.
    distances = cosine_distances(X)

    for i, dataset in enumerate(names):
        nearest = np.argsort(distances[i])

        nearest = [
            j for j in nearest if j != i
        ][:5]

        for rank, j in enumerate(nearest, 1):
            neighbor_records.append({
                "representation": config_name,
                "dataset": dataset,
                "rank": rank,
                "neighbor": names[j],
                "cosine_distance": float(
                    distances[i, j]
                ),
            })

pd.DataFrame(neighbor_records).to_csv(
    OUT_DIR / "nearest_neighbors.csv",
    index=False,
)

# ============================================================
# PCA VISUALIZATIONS
# ============================================================

for config_name, X in processed.items():
    labels = assignments[config_name].to_numpy()

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        s=65,
        alpha=0.85,
    )

    ax.set_title(
        f"{config_name} | K={MAIN_K}"
    )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    )

    ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / f"pca_{config_name}.png",
        dpi=180,
    )

    plt.close(fig)

# ============================================================
# SUMMARY
# ============================================================

k8 = metrics[metrics["k"] == MAIN_K].copy()

summary = k8.merge(
    stability,
    on="representation",
)

summary = summary.sort_values(
    "silhouette",
    ascending=False,
)

summary.to_csv(
    OUT_DIR / "summary_k8.csv",
    index=False,
)

print("\n" + "=" * 75)
print("K=8 COMPARISON")
print("=" * 75)

print(
    summary[
        [
            "representation",
            "silhouette",
            "smallest_cluster",
            "largest_cluster",
            "singleton_clusters",
            "seed_ari_mean",
            "subsample_ari_mean",
        ]
    ].round(3).to_string(index=False)
)

print("\nSaved results to:", OUT_DIR)
print("Completed successfully.")
