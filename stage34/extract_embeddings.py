"""Replicate teammate's 512D mean embedding from TabICLv2 row_interactor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tabicl import TabICLClassifier
from sklearn.model_selection import train_test_split
from stage34.common import read_manifest, read_dataset


def embed(X, y, *, context_rows=256, seed=42, device='cpu'):
    if len(X) <= context_rows:
        raise ValueError(f'Need >{context_rows} rows, found {len(X)}')
    ctx, other = train_test_split(np.arange(len(y)), train_size=context_rows,
                                  stratify=y, random_state=seed)
    model = TabICLClassifier(n_estimators=1, batch_size=1,
             checkpoint_version='tabicl-classifier-v2-20260212.ckpt',
             random_state=seed, device=device, use_amp=(device != 'cpu'), verbose=False)
    model.fit(X[ctx], y[ctx]); model.model_.eval()
    captured = []
    def hook(mod, ins, out):
        captured.append(out.detach().cpu().float().clone())
    handle = model.model_.row_interactor.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.predict_proba(X[other[:1]])
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(f'Expected 1 hook call, got {len(captured)}')
    H = captured[0]
    if H.ndim != 3 or H.shape[:2] != (1, context_rows + 1) or H.shape[-1] != 512:
        raise ValueError(f'Unexpected row representation shape {tuple(H.shape)}')
    return H[0, :context_rows].mean(dim=0).numpy().astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', default='outputs/stage32/embeddings')
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-datasets', type=int, default=0)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    if a.max_datasets: rows = rows[:a.max_datasets]
    failures = []
    for i,r in enumerate(rows,1):
        name = r['dataset']; target = out / (name.replace('/','__') + '.npy')
        if target.is_file():
            print(f'[{i}/{len(rows)}] CACHED {name}', flush=True); continue
        try:
            X,y = read_dataset(a.data_dir, name)
            z = embed(X,y,device=a.device)
            if not np.isfinite(z).all(): raise ValueError('Nonfinite embedding')
            np.save(target,z)
            print(f'[{i}/{len(rows)}] {name} => {z.shape}',flush=True)
        except Exception as e:
            failures.append({'dataset':name,'error':str(e)})
            print(f'[{i}/{len(rows)}] ERROR {name}: {e}',flush=True)
        (out/'failures.json').write_text(json.dumps(failures,indent=2))
    print('Finished. Check failures.json',flush=True)

if __name__=='__main__': main()
