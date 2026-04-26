# OFC 2026 ML Challenge — EDFA Digital Twin

Experiment-running cheat sheet for the EDFA gain-spectrum prediction project.
Architecture / methodology details live in [`docs/experiments.md`](docs/experiments.md); this README is purely about **how to run things**.

---

## 1. Setup

```bash
conda create -n ofc_ml python=3.12 -y
conda activate ofc_ml
pip install -r requirements.txt
pip install matplotlib jupyter          # notebook extras
```

Data layout expected by every script:

```
data/ofc-2026-ml-challenge/
    train_features_clean_1.csv
    train_labels_clean_1.csv
    test_features.csv
    test_labels.csv              # post-competition ground truth
data/cosmos-as-kaggle/
    train_features.csv
    train_labels.csv
```

If `data/cosmos-as-kaggle/` is missing, regenerate it from the COSMOS repo:

```bash
git clone https://github.com/functions-lab/COSMOS-EDFA-Dataset.git COSMOS-EDFA-Dataset
python scripts/cosmos_to_kaggle.py \
    --cosmos-dataset-dir COSMOS-EDFA-Dataset/dataset \
    --out-dir data/cosmos-as-kaggle \
    --category cosmos --gains 18dB --channel-types fix
```

---

## 2. Output layout (multi-seed)

Everything under `results/` is per-seed:

```
results/
    seed_42/
        _pretrain_cache/                 # reusable pretrain checkpoints (by model/config hash)
        _run_state.json                  # run_matrix state for seed 42
        _run_matrix.log                  # append-only log
        m1_ours/
            metrics.json                 # evaluation + per-epoch histories (dB units)
            history.csv                  # long format (stage, epoch, train, val, lr)
            submission.csv               # aligned test-set predictions
            model.pt                     # final weights
            pretrain.pt                  # per-experiment pretrain snapshot (resumable)
            config.snapshot.yaml         # fully-resolved config (incl. seed)
            config.source.yaml           # original YAML copy
            logs/
        ...
    seed_43/ …
    _tables/                             # aggregated paper tables (cross-seed)
        table{1..3}_<group>_per_seed.csv     # long: one row per (seed, experiment)
        table{1..3}_<group>_summary.csv      # mean ± std per experiment across seeds
        figure3_data_scale_*.csv
```

`results/` is gitignored. `_pretrain_cache/` automatically dedupes pretraining across experiments that share the same model config, seed, and data subsample.

---

## 3. The 7 useful commands

### 3.1 Run the whole matrix on one seed

```bash
python scripts/run_matrix.py --seed 42
```

- Runs the 8 canonical experiments in the order defined at the top of `scripts/run_matrix.py` (lightest first, `m4_transformer` last).
- Skips any experiment whose `results/seed_42/<exp>/metrics.json` already exists.
- State is checkpointed in `results/seed_42/_run_state.json` after every experiment, so Ctrl+C is safe — re-run the same command to resume.

Useful extras:

```bash
python scripts/run_matrix.py --seed 42 --dry-run                      # show plan + status
python scripts/run_matrix.py --seed 42 --only m1_ours a_p1_predict_absolute
python scripts/run_matrix.py --seed 42 --force                        # re-run everything
python scripts/run_matrix.py --seed 42 --include-data-scale           # add the 6 data-scale ablations
python scripts/run_matrix.py --seed 42 --override pretrain.epochs=40 finetune.epochs=80
```

### 3.2 Run the matrix across multiple seeds

```bash
python scripts/run_seeds.py --seeds 42 43 44 45 46
```

- Serialises over seeds; each seed is an independent `run_matrix.py --seed S` call.
- Each seed gets its own checkpoint-resume state, so interrupting and re-running the outer command picks up mid-sweep.
- Add `--aggregate-after` to regenerate `results/_tables/` when the sweep finishes.

All `run_matrix.py` flags pass through:

```bash
python scripts/run_seeds.py --seeds 43 44 45 --only m1_ours a_p1_predict_absolute
python scripts/run_seeds.py --seeds 43 44 45 --stop-on-failure
python scripts/run_seeds.py --seeds 43 44 45 --override pretrain.epochs=60
```

### 3.3 Run a single experiment

```bash
python scripts/run_experiment.py --config experiments/main/m1_ours.yaml --seed 42
python scripts/run_experiment.py --config experiments/ablation/physics/a_p1_predict_absolute.yaml \
                                 --seed 43 --force --override finetune.epochs=120
```

The `--seed` flag:
- overrides `cfg.seed` and `cfg.data.random_state`
- scopes all outputs to `results/seed_<SEED>/<exp_name>/`

### 3.4 Resume a fine-tune from a specific pretrain snapshot

Every pretrain-containing run writes `results/seed_<S>/<exp>/pretrain.pt`. To start a new fine-tune from someone else's pretrain:

```bash
python scripts/run_experiment.py \
    --config experiments/ablation/transfer/a_t2_no_pretrain.yaml \
    --seed 42 \
    --override pretrain_weights_path=results/seed_42/m1_ours/pretrain.pt
```

The runner loads the weights, skips the `pretrain` stage, and proceeds to `finetune` / `joint`.

### 3.5 Aggregate across seeds

```bash
python scripts/aggregate_results.py                   # auto-discover all seed_* dirs
python scripts/aggregate_results.py --seeds 42 43 44  # restrict to a subset
```

Produces three table groups (main / transfer / physics) plus one optional data-scale table. For each group:
- `results/_tables/<group>_per_seed.csv` — long format (one row per seed × experiment)
- `results/_tables/<group>_summary.csv` — one row per experiment with `MAE_dB_mean`, `MAE_dB_std`, `MAE_dB_str = "0.0998 ± 0.0023"`, etc.

All dB columns carry an explicit `_dB` suffix; MSE uses `_dB2`.

### 3.6 Visualise in the notebook

```bash
jupyter notebook notebooks/results_viz.ipynb
```

- §§1–5 inspect a single seed (set `SEED` in the first cell; defaults to the first seed found).
- §6 dumps `_tables/*.csv` inline.
- §7 plots cross-seed bar charts with error bars from `*_summary.csv`.
- §8 overlays per-seed training curves for stability checks.

### 3.7 Sanity / smoke tests

```bash
# Smallest possible end-to-end run (<30s on CPU); proves the pipeline works.
python scripts/run_experiment.py \
    --config experiments/main/m1_ours.yaml --seed 999 --force --no-cache \
    --override pretrain.epochs=2 finetune.epochs=3 pretrain.val_every_n_epochs=1 \
               finetune.val_every_n_epochs=1 data.cosmos_ratio=0.005

# Legacy test scripts (pre-refactor pipeline):
python tests/test_pipeline.py
python test_interleaved.py
```

---

## 4. Typical workflows

### A. Quick single-seed iteration (developing / debugging)

```bash
python scripts/run_matrix.py --seed 42 --only m1_ours
jupyter notebook notebooks/results_viz.ipynb   # §§1-5 only
```

### B. Full five-seed sweep for a paper table

```bash
python scripts/run_seeds.py --seeds 42 43 44 45 46 --aggregate-after
# inspect results/_tables/table*_summary.csv
# or open notebooks/results_viz.ipynb → §§6-8
```

### C. Re-running just one ablation across seeds

```bash
python scripts/run_seeds.py --seeds 42 43 44 --only a_p1_predict_absolute --force
python scripts/aggregate_results.py
```

### D. Running an additional seed without touching existing ones

```bash
python scripts/run_matrix.py --seed 47          # prior seeds untouched
python scripts/aggregate_results.py             # tables now include seed 47
```

---

## 5. Experiment catalogue

Defined under `experiments/` as overlayed YAMLs (`base:` chain). Execute any of these by YAML path or stem with `--only <stem>`.

| YAML | Stem | Purpose |
|------|------|---------|
| `experiments/main/m1_ours.yaml`                            | `m1_ours`              | Ours: FourierKAN + physics residual target + pretrain→finetune |
| `experiments/main/m2_mlp.yaml`                              | `m2_mlp`               | Same-capacity MLP baseline |
| `experiments/main/m3_cnn1d.yaml`                            | `m3_cnn1d`             | 1D CNN baseline |
| `experiments/main/m4_transformer.yaml`                      | `m4_transformer`       | Transformer baseline (channel-as-token) |
| `experiments/ablation/transfer/a_t1_no_finetune.yaml`       | `a_t1_no_finetune`     | Use pretrained weights zero-shot |
| `experiments/ablation/transfer/a_t2_no_pretrain.yaml`       | `a_t2_no_pretrain`     | Kaggle from scratch |
| `experiments/ablation/transfer/a_t3_joint.yaml`             | `a_t3_joint`           | Single stage on COSMOS ∪ Kaggle |
| `experiments/ablation/physics/a_p1_predict_absolute.yaml`   | `a_p1_predict_absolute`| Predict absolute gain, no baseline-residual parameterisation |
| `experiments/ablation/data_scale/pretrain_{25,50,100}.yaml` | `ds_pretrain_<R>`      | Data-scale scan over COSMOS ratio |
| `experiments/ablation/data_scale/finetune_{25,50,100}.yaml` | `ds_finetune_<R>`      | Data-scale scan over Kaggle ratio |

The matrix runner schedules items in an order chosen to maximise pretrain-cache reuse and to defer heavy models (Transformer) to the end; see top of `scripts/run_matrix.py` to adjust.

---

## 6. Configuration override cheat sheet

Any YAML field is overridable from the CLI as `key.subkey=value`:

```bash
python scripts/run_experiment.py --config experiments/main/m1_ours.yaml --seed 42 \
    --override device=cuda:0 pretrain.batch_size=512 finetune.epochs=120 \
               model.dropout=0.1 \
               data.cosmos_ratio=0.5
```

Most common knobs: see [`experiments/base.yaml`](experiments/base.yaml).

---

## 7. Troubleshooting

- **Unexpected stale numbers after config changes**: clear cached pretraining and runner state with `rm -rf results/seed_<S>/_pretrain_cache && rm -f results/seed_<S>/_run_state.json`, then re-run with `--force`.
- **Numbers for one experiment look identical to another**: pretrain cache reused across seeds if you forget `--seed`. Each seed owns a separate `results/seed_<S>/_pretrain_cache/`, so simply always pass `--seed`.
- **Out of memory on MPS/CUDA**: `--override pretrain.batch_size=128 finetune.batch_size=32`.
- **Jupyter can't find the package**: the notebook adds `src/` to `sys.path` from the first cell; if you run scripts from a different CWD, `cd` into the repo root first.
