"""Shared dataset utilities for Strategy 3.2, strictly numerical features."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

MANIFEST = Path(__file__).resolve().parent / 'routing_manifest.csv'

def read_manifest(path=MANIFEST):
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def read_dataset(data_dir, dataset, max_features=None):
    folder = Path(data_dir) / dataset
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing dataset {dataset} at {folder}')
    xfile = folder / 'N_train.npy'
    if not xfile.is_file():
        raise FileNotFoundError(f'Numerical features missing: {xfile}')
    X = np.asarray(np.load(xfile), dtype=np.float64)
    y = np.asarray(np.load(folder / 'y_train.npy', allow_pickle=True)).ravel()
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(f'Invalid data matrix for {dataset}')
    if max_features and X.shape[1] > max_features:
        # Deterministic feature selection; train data only.
        with np.errstate(all='ignore'):
            v = np.nanvar(np.where(np.isfinite(X), X, np.nan), axis=0)
        keep = np.sort(np.argsort(-np.nan_to_num(v, nan=0, posinf=0))[:max_features])
        X = X[:, keep]
    X[~np.isfinite(X)] = np.nan
    with np.errstate(all='ignore'):
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    rr, cc = np.where(np.isnan(X))
    X[rr, cc] = med[cc]
    X = np.clip(X, -1e10, 1e10).astype(np.float32)
    if len(np.unique(y)) < 2 or len(np.unique(y)) > 20:
        raise ValueError('Unsupported number of classes')
    _, y = np.unique(y, return_inverse=True)
    return X, np.asarray(y, dtype=np.int64)

def make_episode(X, y, rows, seed, context_rows=None):
    """Sample stratified context/query rows; require all classes in context."""
    rng = np.random.default_rng(seed)
    nclasses = int(np.max(y)) + 1
    if np.bincount(y, minlength=nclasses).min() < 2:
        raise ValueError('Class has <2 rows')
    n = len(y)
    if context_rows is not None:
        if n <= context_rows:
            raise ValueError('Not enough rows for context plus query')
        ctx, remainder = train_test_split(np.arange(n), train_size=context_rows,
                                          stratify=y, random_state=seed)
        return ctx, remainder
    limit = min(rows, n)
    if limit < n:
        selected, _ = train_test_split(np.arange(n), train_size=limit,
                                      stratify=y, random_state=seed)
    else:
        selected = np.arange(n)
    # A majority of samples as context, some query.
    query_n = max(nclasses, int(len(selected) * 0.25))
    query_n = min(query_n, len(selected) - nclasses)
    if query_n < nclasses:
        raise ValueError('Insufficient support and query sizes')
    ctx, qry = train_test_split(selected, test_size=query_n, stratify=y[selected],
                                random_state=seed + 1)
    if len(np.unique(y[ctx])) != nclasses:
        raise ValueError('Context missing class')
    return ctx, qry
