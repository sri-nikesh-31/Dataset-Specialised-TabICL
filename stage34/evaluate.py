"""Stage 4: nearest-centroid routing and held-out meta-test evaluation."""
import argparse
import csv
import gc
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier
from stage34.common import read_manifest,read_dataset
from stage34.extract_embeddings import embed

METRICS=['accuracy','balanced_accuracy','macro_f1','log_loss']

def infer(path,Xs,ys,Xq,nc,device,seed):
    model=TabICLClassifier(model_path=None if path is None else str(path),
        checkpoint_version='tabicl-classifier-v2-20260212.ckpt',
        n_estimators=1,batch_size=1,device=device,
        use_amp=(device!='cpu'),random_state=seed,verbose=False)
    try:
        model.fit(Xs,ys)
        P=np.zeros((len(Xq),nc),dtype=np.float64)
        P[:,np.asarray(model.classes_,dtype=int)]=model.predict_proba(Xq)
        return P
    finally:
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def measure(y,p):
    n=int(np.max(y))+1
    p=np.clip(p,1e-12,1);p=p/p.sum(axis=1,keepdims=True)
    yp=np.argmax(p,axis=1)
    return {'accuracy':float(accuracy_score(y,yp)),
            'balanced_accuracy':float(balanced_accuracy_score(y,yp)),
            'macro_f1':float(f1_score(y,yp,average='macro',zero_division=0)),
            'log_loss':float(log_loss(y,p,labels=np.arange(n)))}

def write_csv(path,records):
    if not records:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True)
    p.add_argument('--router',default='outputs/stage32/router.joblib')
    p.add_argument('--experts-root',default='outputs/stage32/experts')
    p.add_argument('--output',default='outputs/stage32/evaluation')
    p.add_argument('--device',default='cuda')
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--query',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-datasets',type=int,default=0)
    a=p.parse_args()
    if a.top_k<1 or a.top_k>4:p.error('top-k must be between 1 and 4')
    router=joblib.load(a.router)
    test=[r for r in read_manifest() if r['split']=='meta_test']
    if a.max_datasets:test=test[:a.max_datasets]
    ckpts=[Path(a.experts_root)/f'expert_{j}'/'best.ckpt' for j in range(4)]
    missing=[str(p) for p in ckpts if not p.is_file()]
    if missing:raise SystemExit('Missing checkpoints: '+', '.join(missing))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    results=[];errors=[];selections=[]
    for i,r in enumerate(test,1):
        name=r['dataset']
        try:
            X,y=read_dataset(a.data_dir,name)
            if len(X)<=256:raise ValueError('Need at least 257 rows')
            # Identical 256 context sampling to the teammate's strategy_3_2.py.
            ctx,rem=train_test_split(np.arange(len(y)),train_size=256,
                   stratify=y,random_state=a.seed)
            query_idx=rem[:a.query]
            Xs,ys,Xq,yq=X[ctx],y[ctx],X[query_idx],y[query_idx]
            nc=len(np.unique(y))
            # Embed with pretrained TabICL using only labeled support context.
            z=embed(X,y,context_rows=256,seed=a.seed,device='cpu')
            zz=router['scaler'].transform(z[router['keep']].reshape(1,-1))
            distances=router['model'].transform(zz)[0]
            ranked=np.argsort(distances)
            nearest=int(ranked[0]); top=[int(j) for j in ranked[:a.top_k]]
            predicted_cluster=int(router['model'].predict(zz)[0])
            if predicted_cluster!=nearest:
                raise RuntimeError('Nearest center and KMeans.predict disagree')
            ref=infer(None,Xs,ys,Xq,nc,a.device,a.seed)
            specialist=[infer(ckpts[j],Xs,ys,Xq,nc,a.device,a.seed) for j in top]
            metrics={'baseline':measure(yq,ref),
                     'top1':measure(yq,specialist[0]),
                     'topk':measure(yq,np.mean(specialist,axis=0))}
            rec={'dataset':name,'meta_test':True,'expert_top1':nearest,
                 'topk_experts':json.dumps(top),'n_query':len(yq)}
            for method,values in metrics.items():
                rec.update({f'{method}_{m}':value for m,value in values.items()})
            results.append(rec)
            selections.append({'dataset':name,'experts':top,'distances':[float(distances[j]) for j in top]})
            print(f'[{i}/{len(test)}] {name}: baseline={metrics["baseline"]["accuracy"]:.4f} '
                  f'top1={metrics["top1"]["accuracy"]:.4f} '
                  f'top{a.top_k}={metrics["topk"]["accuracy"]:.4f} chosen={top}',flush=True)
        except KeyboardInterrupt:raise
        except Exception as exc:
            errors.append({'dataset':name,'error':f'{type(exc).__name__}: {exc}'})
            print(f'[{i}/{len(test)}] ERROR {name}: {exc}',flush=True)
        write_csv(out/'per_dataset.csv',results)
        (out/'errors.json').write_text(json.dumps(errors,indent=2))
        (out/'selections.json').write_text(json.dumps(selections,indent=2))
    summary={'n_evaluated':len(results),'n_failed':len(errors),'n_requested':len(test),'top_k':a.top_k}
    for method in ['baseline','top1','topk']:
        if results:summary[method]={m:float(np.mean([r[f'{method}_{m}'] for r in results])) for m in METRICS}
    if results:
        summary['top1_minus_baseline']={m:summary['top1'][m]-summary['baseline'][m] for m in METRICS}
        summary['topk_minus_baseline']={m:summary['topk'][m]-summary['baseline'][m] for m in METRICS}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY:',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
