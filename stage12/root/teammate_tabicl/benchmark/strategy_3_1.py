
import os
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\mahidhar\Projects\DS Project\benchmark")

DATA_DIR = BASE_DIR / "dataset-250115" / "data"

OUTPUT_DIR = BASE_DIR / "representations" / "strategy_3_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Maximum observations used for expensive statistical operations.
# Dataset size is still recorded from the full training data.
MAX_ROWS = 5000

# Maximum feature pairs for correlation statistics.
MAX_CORR_PAIRS = 2000

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(x):
    x = float(x)
    return x if np.isfinite(x) else np.nan


def aggregate(values, prefix):
    """
    Convert variable-length feature statistics into
    fixed-dimensional summary statistics.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
        }

    return {
        f"{prefix}_mean": safe_float(np.mean(values)),
        f"{prefix}_std": safe_float(np.std(values)),
        f"{prefix}_min": safe_float(np.min(values)),
        f"{prefix}_max": safe_float(np.max(values)),
        f"{prefix}_median": safe_float(np.median(values)),
        f"{prefix}_q25": safe_float(np.quantile(values, 0.25)),
        f"{prefix}_q75": safe_float(np.quantile(values, 0.75)),
    }


def sample_rows(X, y, max_rows, seed):
    """
    Deterministic stratified row sampling.
    """
    if len(X) <= max_rows:
        return X, y

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))

    try:
        _, selected = train_test_split(
            indices,
            test_size=max_rows,
            stratify=y,
            random_state=seed,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            indices, size=max_rows, replace=False
        )

    selected = np.sort(selected)

    return X[selected], y[selected]


# ============================================================
# DATASET META-FEATURE EXTRACTION
# ============================================================

def extract_meta_features(X, y):
    """
    Extract one fixed-dimensional statistical representation
    of an entire labelled numerical binary dataset.
    """

    features = {}

    n_rows, n_cols = X.shape

    # --------------------------------------------------------
    # 1. DATASET SIZE / DIMENSIONALITY
    # --------------------------------------------------------

    features["log_n_rows"] = np.log1p(n_rows)
    features["log_n_features"] = np.log1p(n_cols)
    features["feature_row_ratio"] = n_cols / max(n_rows, 1)

    # --------------------------------------------------------
    # 2. MISSINGNESS / NUMERICAL QUALITY
    # --------------------------------------------------------

    X = np.asarray(X, dtype=np.float64)

    finite_mask = np.isfinite(X)

    features["missing_fraction"] = safe_float(
        1.0 - np.mean(finite_mask)
    )

    missing_per_feature = 1.0 - np.mean(
        finite_mask, axis=0
    )

    features.update(
        aggregate(missing_per_feature, "missing_feature")
    )

    features["features_with_missing_fraction"] = safe_float(
        np.mean(missing_per_feature > 0)
    )

    # Replace nonfinite values with NaN.
    X_clean = X.copy()
    X_clean[~finite_mask] = np.nan

    # --------------------------------------------------------
    # 3. FEATURE DISTRIBUTION STATISTICS
    # --------------------------------------------------------

    means = []
    stds = []
    skews = []
    kurtoses = []
    ranges = []
    unique_ratios = []
    zero_fractions = []

    constant_count = 0

    for j in range(n_cols):

        col = X_clean[:, j]
        col = col[np.isfinite(col)]

        if len(col) == 0:
            constant_count += 1
            continue

        col_mean = np.mean(col)
        col_std = np.std(col)

        means.append(col_mean)
        stds.append(col_std)
        ranges.append(np.max(col) - np.min(col))

        unique_ratios.append(
            len(np.unique(col)) / len(col)
        )

        zero_fractions.append(np.mean(col == 0))

        if col_std < 1e-12:
            constant_count += 1
            skews.append(0.0)
            kurtoses.append(0.0)
        else:
            skews.append(
                safe_float(skew(col, bias=False))
            )
            kurtoses.append(
                safe_float(kurtosis(col, bias=False))
            )

    features["constant_feature_fraction"] = (
        constant_count / n_cols
    )

    for name, values in [
        ("feature_mean", means),
        ("feature_std", stds),
        ("feature_skew", skews),
        ("feature_kurtosis", kurtoses),
        ("feature_range", ranges),
        ("unique_ratio", unique_ratios),
        ("zero_fraction", zero_fractions),
    ]:
        features.update(aggregate(values, name))

    # --------------------------------------------------------
    # 4. TARGET DISTRIBUTION
    # --------------------------------------------------------

    classes, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()

    features["n_classes"] = len(classes)
    features["class_entropy"] = safe_float(
        entropy(probabilities)
    )

    features["minority_class_fraction"] = safe_float(
        np.min(probabilities)
    )

    features["class_imbalance_ratio"] = safe_float(
        np.max(counts) / max(np.min(counts), 1)
    )

    # --------------------------------------------------------
    # 5. IMPUTE FOR DEPENDENCY ANALYSIS
    # --------------------------------------------------------

    # Median imputation is applied only to the sampled
    # training observations for dependency calculations.
    X_imputed = X_clean.copy()

    for j in range(n_cols):
        col = X_imputed[:, j]
        valid = np.isfinite(col)

        fill = np.median(col[valid]) if valid.any() else 0.0
        col[~valid] = fill

    # --------------------------------------------------------
    # 6. FEATURE-FEATURE CORRELATIONS
    # --------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    if n_cols >= 2:

        pairs = [
            (i, j)
            for i in range(n_cols)
            for j in range(i + 1, n_cols)
        ]

        if len(pairs) > MAX_CORR_PAIRS:
            selected = rng.choice(
                len(pairs),
                size=MAX_CORR_PAIRS,
                replace=False
            )
            pairs = [pairs[i] for i in selected]

        correlations = []

        for i, j in pairs:
            a = X_imputed[:, i]
            b = X_imputed[:, j]

            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue

            corr = np.corrcoef(a, b)[0, 1]

            if np.isfinite(corr):
                correlations.append(abs(corr))

    else:
        correlations = []

    features.update(
        aggregate(correlations, "absolute_correlation")
    )

    features["high_correlation_fraction"] = (
        safe_float(np.mean(np.array(correlations) > 0.8))
        if correlations else 0.0
    )

    # --------------------------------------------------------
    # 7. FEATURE-TARGET CORRELATIONS
    # --------------------------------------------------------

    y_encoded = np.unique(y, return_inverse=True)[1]

    target_correlations = []

    for j in range(n_cols):

        col = X_imputed[:, j]

        if np.std(col) < 1e-12:
            continue

        corr = np.corrcoef(col, y_encoded)[0, 1]

        if np.isfinite(corr):
            target_correlations.append(abs(corr))

    features.update(
        aggregate(target_correlations, "target_correlation")
    )

    # --------------------------------------------------------
    # 8. FEATURE-TARGET MUTUAL INFORMATION
    # --------------------------------------------------------

    try:
        mi = mutual_info_classif(
            X_imputed,
            y_encoded,
            discrete_features=False,
            random_state=RANDOM_SEED,
            n_neighbors=3,
        )

        features.update(
            aggregate(mi, "mutual_information")
        )

    except Exception:
        features.update(
            aggregate([], "mutual_information")
        )

    return features


# ============================================================
# PROCESS ALL DATASETS
# ============================================================

dataset_names = []
feature_records = []
failures = []

dataset_folders = sorted(
    p for p in DATA_DIR.iterdir() if p.is_dir()
)

print("=" * 80)
print("STRATEGY 3.1 - STATISTICAL DATASET REPRESENTATION")
print("=" * 80)
print("Data directory:", DATA_DIR)
print("Total dataset folders:", len(dataset_folders))

for folder in tqdm(dataset_folders):

    try:
        info_path = folder / "info.json"

        if not info_path.exists():
            continue

        with open(info_path, "r") as f:
            info = json.load(f)

        if info.get("task_type") != "binclass":
            continue

        if info.get("n_cat_features") != 0:
            continue

        x_path = folder / "N_train.npy"
        y_path = folder / "y_train.npy"

        if not x_path.exists() or not y_path.exists():
            continue

        X = np.load(x_path, allow_pickle=False)
        y = np.load(y_path, allow_pickle=True)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid X or y shape")

        if len(X) != len(y):
            raise ValueError("X/y length mismatch")

        if len(np.unique(y)) != 2:
            raise ValueError("Expected exactly two classes")

        X_sample, y_sample = sample_rows(
            X, y, MAX_ROWS, RANDOM_SEED
        )

        meta = extract_meta_features(
            X_sample, y_sample
        )

        # Preserve actual dataset dimensions rather than
        # sampled dimensions.
        meta["log_n_rows"] = np.log1p(X.shape[0])
        meta["log_n_features"] = np.log1p(X.shape[1])
        meta["feature_row_ratio"] = (
            X.shape[1] / X.shape[0]
        )

        dataset_names.append(folder.name)
        feature_records.append(meta)

    except Exception as e:

        failures.append({
            "dataset": folder.name,
            "error": str(e)
        })

# ============================================================
# BUILD EMBEDDING MATRIX
# ============================================================

if not feature_records:
    raise RuntimeError(
        "No datasets processed. Check DATA_DIR and file structure."
    )

feature_names = sorted(
    set().union(*(r.keys() for r in feature_records))
)

Z_raw = np.array(
    [
        [record.get(name, np.nan) for name in feature_names]
        for record in feature_records
    ],
    dtype=np.float64
)

# Replace infinities with NaN before imputation.
Z_raw[~np.isfinite(Z_raw)] = np.nan

# Handle any meta-features missing for all datasets.
all_missing = np.all(np.isnan(Z_raw), axis=0)
Z_raw[:, all_missing] = 0.0

imputer = SimpleImputer(strategy="median")
Z_imputed = imputer.fit_transform(Z_raw)

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z_imputed)

# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    OUTPUT_DIR / "statistical_embeddings_raw.npy",
    Z_raw
)

np.save(
    OUTPUT_DIR / "statistical_embeddings_scaled_exploratory.npy",
    Z_scaled
)

with open(OUTPUT_DIR / "dataset_names.json", "w") as f:
    json.dump(dataset_names, f, indent=2)

with open(OUTPUT_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

with open(OUTPUT_DIR / "failures.json", "w") as f:
    json.dump(failures, f, indent=2)

with open(OUTPUT_DIR / "scaler_parameters.json", "w") as f:
    json.dump(
        {
            "imputer_statistics": imputer.statistics_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist()
        },
        f,
        indent=2
    )

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print("Processed datasets:", len(dataset_names))
print("Failed datasets:", len(failures))
print("Raw embedding shape:", Z_raw.shape)
print("Scaled embedding shape:", Z_scaled.shape)

print("NaN in scaled embeddings:", np.isnan(Z_scaled).any())
print("Inf in scaled embeddings:", np.isinf(Z_scaled).any())

print("\nOutput directory:")
print(OUTPUT_DIR)

if failures:
    print("\nFailures:")
    for failure in failures:
        print(failure)
