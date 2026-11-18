"""Stage 3: multi-dataset episodic TabICLv2 fine-tuning, one cluster per run."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from tabicl import FinetunedTabICLClassifier
from tabicl._finetune.data import iter_epoch_meta_batches, move_meta_batch
from stage34.common import read_manifest,read_dataset,make_episode


def get_batch(X,y,rows,features,seed):
    ctx,qry=make_episode(X,y,rows,seed)
    selected=np.concatenate([ctx,qry]); x=X[selected]; labels=y[selected]
    if x.shape[1]>features:
        rng=np.random.default_rng(seed)
        feats=np.sort(rng.choice(x.shape[1],size=features,replace=False))
        x=x[:,feats]
    # Force exact context/query row assignment via upstream preprocessor's
    # chunk generator, while preserving sklearn class-label preprocessing.
    iterator=iter_epoch_meta_batches(
        x,labels,classification=True,n_estimators=1,
        max_chunk_size=len(labels),query_ratio=0.25,
        epoch_seed=seed,preprocessing_seed=seed,
        norm_methods=None,feat_shuffle_method='latin',
        class_shuffle_method='shift',outlier_threshold=4.0,
        min_chunk_size=1,
    )
    return next(iterator)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True)
    p.add_argument('--expert',type=int,required=True,choices=range(4))
    p.add_argument('--output',default='outputs/stage32/experts')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--steps-per-epoch',type=int,default=30)
    p.add_argument('--rows',type=int,default=64)
    p.add_argument('--features',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--val-fraction',type=float,default=0.2)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA is required for this training configuration')
    if a.epochs<1 or a.steps_per_epoch<1: p.error('steps/epochs must be positive')
    torch.manual_seed(a.seed+a.expert)
    np.random.seed(a.seed+a.expert)
    dev=torch.device('cuda')
    rows=[r for r in read_manifest() if r['split']=='meta_train' and int(r['tab_mean_standard_cluster'])==a.expert]
    if len(rows)<3: raise ValueError('Too few training tasks')
    # Split by dataset family, not by individual rows. Never use meta_test.
    groups=np.asarray([r['family'] for r in rows]); names=np.arange(len(rows))
    splitter=GroupShuffleSplit(n_splits=1,test_size=a.val_fraction,random_state=a.seed)
    tr,va=next(splitter.split(names,groups=groups))
    train_names=[rows[i]['dataset'] for i in tr]
    val_names=[rows[i]['dataset'] for i in va]
    out=Path(a.output)/f'expert_{a.expert}';out.mkdir(parents=True,exist_ok=True)
    train=[];val=[];failed=[]
    for name,where in [(n,train) for n in train_names]+[(n,val) for n in val_names]:
        try:
            X,y=read_dataset(a.data_dir,name)
            if len(X)<max(a.rows,64) or np.bincount(y).min()<3:
                raise ValueError('Insufficient class/row support')
            where.append((name,X,y))
        except Exception as e: failed.append({'dataset':name,'error':str(e)})
    print(f'expert={a.expert} candidate_train={len(train_names)} candidate_val={len(val_names)} '
          f'usable_train={len(train)} usable_val={len(val)}',flush=True)
    if not train or not val: raise SystemExit(f'Cannot train: missing usable data. Failed: {failed}')
    ft=FinetunedTabICLClassifier(
        device='cuda',n_estimators_finetune=1,n_estimators_validation=1,
        n_estimators_inference=1,freeze_col=True,freeze_row=True,freeze_icl=False,
        amp=True, learning_rate=a.learning_rate,early_stopping=False,
        random_state=a.seed,verbose=False)
    ft._load_pretrained(dev)
    ft.model_.to(dev)
    ft._apply_freezing(ft.model_)
    params=[q for q in ft.model_.parameters() if q.requires_grad]
    print('Trainable parameters:',sum(q.numel() for q in params),flush=True)
    optimizer=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    best=float('inf');history=[]

    def sample_loss(item,seed,training):
        _,X,y=item
        b=get_batch(X,y,a.rows,a.features,seed)
        b=move_meta_batch(b,dev)
        if ft._task_skip_batch(b):
            raise ValueError('Query label missing from context')
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
            loss=ft._compute_batch_loss(b,ft.model_)
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        return loss

    def validate(epoch):
        ft._set_training_mode(False)
        losses=[]
        with torch.no_grad():
            for j,item in enumerate(val):
                try: losses.append(float(sample_loss(item,500000+j+a.seed,False).item()))
                except (ValueError,RuntimeError) as e:
                    print(f'val SKIP {item[0]}: {e}',flush=True)
        return float(np.mean(losses)) if losses else float('inf'),len(losses)

    # Select best checkpoint against unchanged pretrained initialisation.
    init_val,nval=validate(0)
    print('Pretrained validation loss:',init_val,'tasks:',nval,flush=True)
    def save(epoch,metric,path):
        state={'config':ft.model_config_,
               'state_dict':{k:v.detach().cpu().clone() for k,v in ft.model_.state_dict().items()},
               'curr_step':epoch,'epoch':epoch,'best_metric':metric}
        torch.save(state,path)
    if np.isfinite(init_val):
        best=init_val
        save(0,best,out/'best.ckpt')
    rng=np.random.default_rng(a.seed+1000+a.expert)
    for epoch in range(1,a.epochs+1):
        ft._set_training_mode(True)
        accum=[];skip=0
        for step in range(a.steps_per_epoch):
            ok=False
            for retry in range(20):
                item=train[int(rng.integers(0,len(train)))]
                seed=int(rng.integers(0,2**31-1))
                try:
                    loss=sample_loss(item,seed,True)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params,1.0)
                    scaler.step(optimizer);scaler.update()
                    accum.append(float(loss.item()));ok=True;break
                except (ValueError,RuntimeError) as exc:
                    optimizer.zero_grad(set_to_none=True)
                    if 'out of memory' in str(exc).lower():
                        torch.cuda.empty_cache()
                    if retry==19: print('Update SKIP:',exc,flush=True)
            if not ok:skip+=1
        vl,count=validate(epoch)
        mean_train=float(np.mean(accum)) if accum else None
        is_best=np.isfinite(vl) and vl<best
        if is_best:
            best=vl;save(epoch,best,out/'best.ckpt')
        rec={'epoch':epoch,'training_loss':mean_train,'validation_loss':vl,
             'updates':len(accum),'skips':skip,'validation_tasks':count,'is_best':bool(is_best)}
        history.append(rec)
        print(f'epoch {epoch}: train={mean_train} val={vl:.4f} updates={len(accum)} '
              f'skips={skip} val_tasks={count} best={is_best}',flush=True)
    save(a.epochs,vl,out/'last.ckpt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'settings.json').write_text(json.dumps(vars(a)|{
         'train_datasets':[x[0] for x in train],
         'validation_datasets':[x[0] for x in val],
         'failures':failed,'pretrained_val_loss':init_val},indent=2))
    print('Saved:',out/'best.ckpt',flush=True)

if __name__=='__main__':main()
