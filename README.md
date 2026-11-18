# Dataset-Specialised TabICL — Strategy 3.2, Stages 3 and 4

This is a **prototype** built from your teammate's repository ZIP, with its committed `routing_manifest.csv`. It has been syntax-checked but not GPU-tested against your local TabICL checkpoint, so first use the small smoke test. Requires TabICL v2 with internal `FinetunedTabICLClassifier` APIs and an available pretrained checkpoint. No model weights or source datasets are included.

## Data required

The original **dataset-250115/data** directory, with `N_train.npy` and `y_train.npy` in each named dataset directory. Your prior Windows datasets may already be at `/mnt/d/DS_Proj/dataset-250115/data`. This prototype intentionally replicates the teammate's **numerical-only** Strategy 3.2. The original strategy's lack of categorical-feature support is not repaired here because that would change embedding semantics and clustering.

## Install into your WSL project

```bash
cd ~
unzip -o stage34_tabicl_k4.zip
cd ~/hite3050_revised
cp -r ~/stage34_tabicl_k4/stage34 .
source ~/hite3050/.venv/bin/activate
export DATA=/mnt/d/DS_Proj/dataset-250115/data
```

## 1. Regenerate all 512D representations and train-only router

The teammate repository ZIP contains **no embedding .npy files and no fitted router .joblib**; all 67 numerical datasets must be available to reproduce KMeans.

```bash
python -m stage34.extract_embeddings --data-dir "$DATA" --device cpu
python -m stage34.build_router
```

Embedding extraction recreates the teammate's 256-row stratified context sampling + forward hook on `row_interactor`; it is CPU inference by default as in the original script. Results cached independently per dataset so interrupted runs can resume. `build_router` checks that regenerated train labels match the original committed CSV cluster assignments. **If they don't match, do not train or route using those new assignments without reconciling them.** The `--allow-different-assignments` switch exists for diagnosis only and is not a substitute for reassigning/retraining the experts.

## 2. Stage 3 smoke-test expert 0

```bash
python -m stage34.train_experts --data-dir "$DATA" --expert 0 --epochs 1 --steps-per-epoch 5 --rows 64 --features 32
```

This trains the ICL predictor; column and row modules are frozen. It holds out some meta-training task families for expert checkpoint selection; **none of the 14 meta-test datasets are used**. Validation episodes use internally reproducible seeds and pretrained validation loss serves as best-checkpoint reference.

## 3. Stage 3 all four experts

```bash
for i in 0 1 2 3; do
  python -m stage34.train_experts --data-dir "$DATA" --expert "$i" --epochs 3 --steps-per-epoch 30 --rows 64 --features 32 || break
done
```

**Warning:** Running the full command for expert 0 overwrites its smoke-test checkpoint. Separate the smoke output directory if keeping it is important. On an RTX 3050 6GB only train/evaluate one expert at a time.

## 4. Stage 4 smoke-test then evaluate 14 meta-test datasets

```bash
python -m stage34.evaluate --data-dir "$DATA" --max-datasets 2 --top-k 3
python -m stage34.evaluate --data-dir "$DATA" --max-datasets 0 --top-k 3
cat outputs/stage32/evaluation/summary.json
```

The evaluator extracts 512D TabICL representations with the *original pretrained model*, routes via the saved fitted KMeans, and compares pretrained TabICL, Top-1 expert and uniform mean-probability Top-3 ensemble on the **same test query rows**. The representation depends exclusively on the 256 labelled context rows, not query labels. In keeping with the original Stage 1 code, evaluation samples support/query from `N_train.npy`, rather than using benchmark `N_test.npy`; this is a held-out-row evaluation within meta-test tasks, not the original benchmark test partition.

## Important experimental boundaries

- The benchmark split (53/14) and K=4 assignments come from teammate `routing_manifest.csv`, column `tab_mean_standard_cluster`. The router must be regenerated from original embeddings using the same TabICL checkpoint and preprocessing.
- All features are *numerical only*, matching the teammate's original 512D extractor.
- `train_experts.py` relies on TabICL private fine-tuning methods and might require minor adaptation if your installation differs. Test a five-step run first.
- Experts train on subsets of the 53 meta-train datasets because we reserve family-aware validation tasks **within each cluster**. This is more conservative than using all 53 for optimization. Be explicit in reports about the held-out validation task identities.
- **The meta-test dataset embeddings are used for routing, not fitting KMeans or selecting checkpoints.** They must never be used for hyperparameter selection.
- To estimate variance and avoid overclaiming, repeat evaluation over multiple fixed random splits/seeds after the first run, without using the test sets to tune training.
- Actual specialist competence may not improve upon pretrained TabICL, even when the representation and routing work.

## Project Structure

- stage12/: Original dataset representation learning,
  embedding stability experiments and dataset clustering.
- stage34/: Multi-dataset expert fine-tuning,
  retrieval-based prediction and routing ablations.
- results/: Recorded experimental results.

## Acknowledgements

The original Stage 1 and Stage 2 implementation was
developed by Mahidhar as part of our team project.

Original repository:
https://github.com/Mahidhar21/Dataset-Specialised-TabICL

Stage 3 and Stage 4 extend this implementation with
episodic TabICL v2 specialist training and evaluation.
