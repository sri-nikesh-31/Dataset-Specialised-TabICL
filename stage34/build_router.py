"""Fit K=4 train-only scaler and KMeans on recreated 512D embeddings."""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from stage34.common import read_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--embeddings',default='outputs/stage32/embeddings')
    p.add_argument('--output',default='outputs/stage32/router.joblib')
    p.add_argument('--allow-different-assignments',action='store_true',
                   help='For exploratory runs only; this invalidates matching the teammate cluster IDs')
    a=p.parse_args()
    rows=read_manifest(); names=[r['dataset'] for r in rows]
    Z=np.stack([np.load(Path(a.embeddings)/(n.replace('/','__')+'.npy')) for n in names])
    if Z.shape!=(len(names),512): raise ValueError(f'Unexpected embeddings {Z.shape}')
    train=np.asarray([i for i,r in enumerate(rows) if r['split']=='meta_train'])
    test=np.asarray([i for i,r in enumerate(rows) if r['split']=='meta_test'])
    keep=np.var(Z[train],axis=0)>1e-12
    scaler=StandardScaler(); Xt=scaler.fit_transform(Z[train][:,keep]);
    model=KMeans(n_clusters=4, random_state=42, n_init=20)
    pred=model.fit_predict(Xt)
    expected=np.asarray([int(rows[i]['tab_mean_standard_cluster']) for i in train])
    mismatches=int(np.sum(pred!=expected))
    print('Training-cluster assignments differ from committed CSV:',mismatches,'/',len(train))
    if mismatches and not a.allow_different_assignments:
        raise SystemExit('ROUTER NOT SAVED: do not use different KMeans labels with teammates checkpoints. '
           'Verify the data source, TabICL version, preprocessing and CPU extraction.')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'keep':keep,'scaler':scaler,'model':model,'seed':42,
                 'representation':'TabICL 512D mean row_interactor'},out)
    print('Saved router:',out)
    print('Train sizes:',np.bincount(pred,minlength=4).tolist())
    print('Test sizes:',np.bincount(model.predict(scaler.transform(Z[test][:,keep])),minlength=4).tolist())

if __name__=='__main__':main()
