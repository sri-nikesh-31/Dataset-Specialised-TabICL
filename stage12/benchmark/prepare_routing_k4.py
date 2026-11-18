
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

BASE = Path(r"C:\mahidhar\Projects\DS Project\benchmark")
STAT = BASE / "representations" / "strategy_3_1"
TAB = BASE / "representations" / "strategy_3_2"
OUT = BASE / "representations" / "routing_k4"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 4

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

names = read_json(STAT / "dataset_names.json")
tab_names = read_json(TAB / "dataset_names.json")

assert len(names) == len(set(names))
assert len(tab_names) == len(set(tab_names))
assert set(names) == set(tab_names)

lookup = {name: i for i, name in enumerate(tab_names)}
order = [lookup[name] for name in names]

data = {
    "stat_signed_log": np.load(
        STAT / "statistical_embeddings_raw.npy"
    ),
    "tab_mean_standard": np.load(
        TAB / "tabicl_mean_embeddings.npy"
    )[order],
    "tab_mean_std_standard": np.load(
        TAB / "tabicl_mean_std_embeddings.npy"
    )[order],
}

def family(name):
    lower = name.lower()

    if lower.startswith("forex_"):
        return "FOREX"

    if lower.startswith("contaminant-detection-in-packaged"):
        return "microwave_sensing"

    if lower in {"ada", "ada_agnostic"}:
        return "ada"

    # Conservative domain-level grouping.
    if lower in {"pc1", "pc3", "pc4", "kc1", "jm1"}:
        return "software_defect"

    return "individual::" + name

groups = np.array([family(n) for n in names])

# Fixed, reproducible group-aware split.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_idx, test_idx = next(
    splitter.split(np.zeros(len(names)), groups=groups)
)

assert set(groups[train_idx]).isdisjoint(
    set(groups[test_idx])
)

manifest = pd.DataFrame({
    "dataset": names,
    "family": groups,
    "split": [
        "meta_train" if i in set(train_idx) else "meta_test"
        for i in range(len(names))
    ],
})

summaries = []

for config, matrix in data.items():
    X = np.asarray(matrix, dtype=np.float64)

    if X.ndim != 2 or X.shape[0] != len(names):
        raise ValueError(f"Bad shape: {config}")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values: {config}")

    if config == "stat_signed_log":
        X = np.sign(X) * np.log1p(np.abs(X))

    keep = np.var(X[train_idx], axis=0) > 1e-12
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(X[train_idx][:, keep])
    Xtest = scaler.transform(X[test_idx][:, keep])

    model = KMeans(
        n_clusters=K,
        random_state=SEED,
        n_init=20
    )

    train_labels = model.fit_predict(Xtrain)
    test_labels = model.predict(Xtest)

    labels = np.full(len(names), -1, dtype=int)
    labels[train_idx] = train_labels
    labels[test_idx] = test_labels

    manifest[config + "_cluster"] = labels

    counts = np.bincount(train_labels, minlength=K)
    print(config, "train cluster sizes:", counts.tolist())

    for cluster in range(K):
        summaries.append({
            "representation": config,
            "cluster": cluster,
            "n_meta_train": int(counts[cluster]),
            "n_meta_test": int(
                np.sum(test_labels == cluster)
            ),
        })

    joblib.dump({
        "selected_dimensions": keep,
        "scaler": scaler,
        "kmeans": model,
        "signed_log": config == "stat_signed_log",
    }, OUT / f"router_{config}.joblib")

manifest.to_csv(OUT / "routing_manifest.csv", index=False)

pd.DataFrame(summaries).to_csv(
    OUT / "routing_cluster_summary.csv",
    index=False
)

manifest[["dataset", "family", "split"]].to_csv(
    OUT / "dataset_families.csv",
    index=False
)

print("\nMeta-train:", len(train_idx))
print("Meta-test:", len(test_idx))

print("\nHeld-out datasets:")
print(
    manifest.loc[
        manifest["split"] == "meta_test",
        ["dataset", "family"]
    ].to_string(index=False)
)

print("\nOutput:", OUT)
print("Completed.")
