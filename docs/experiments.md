# EDFA Digital Twin — Experiments

This document is the paper-ready "Experiments" chapter: problem setup, datasets,
evaluation protocol, and the full matrix (Main results + Ablations) together
with the scripts used to produce them.

All numbers are generated automatically by

```bash
python scripts/run_all.py                    # run every experiments/**/*.yaml
python scripts/aggregate_results.py          # build tables under results/_tables/
```

and the tables referenced below live in `[results/_tables/](../results/_tables)`.

---

## 1. Experimental setup

### 1.1 Task

Given a single EDFA measurement, predict the per-channel **gain spectrum**
over 95 C-band WDM channels. The reference ground truth is the log-domain
ratio between the EDFA output and input power spectra.

### 1.2 Datasets


| Role          | Source                                                                      | #samples | Used by stage        |
| ------------- | --------------------------------------------------------------------------- | -------- | -------------------- |
| Pre-training  | COSMOS-EDFA (open-source) converted to Kaggle schema                        | 41 440   | `pretrain` / `joint` |
| Fine-tuning   | OFC 2026 ML Challenge Kaggle train (cleaned, `train_features_clean_1.csv`)  | 473      | `finetune` / `joint` |
| Held-out test | OFC 2026 ML Challenge Kaggle test (with post-competition `test_labels.csv`) | 21 056   | offline evaluation   |


The Kaggle train set is intentionally tiny (~0.5k rows), making the task a
**low-resource transfer learning** problem: most of the physical regularities
must be learned from the large COSMOS corpus. The test set is split into two
halves by the `Usage` column (Public / Private), matching the original Kaggle
leaderboard.

### 1.3 Feature preprocessing

Preprocessing follows `[src/ofc_ml/features.py](../src/ofc_ml/features.py)`:

1. dBm → linear mW on the 95 `EDFA_input_spectra_`* channels
2. `StandardScaler` on 4 numerical scalars (`target_gain`,
  `target_gain_tilt`, `EDFA_input_power_total`,
   `EDFA_output_power_total`)
3. One-hot encoding of `EDFA_type` and `edfa_index`
4. Pass-through of the 95-dim `DUT_WSS_activated_channel_index_*` mask
  (USE_MASK=concat)
5. The preprocessor is fit **once** on COSMOS ∪ Kaggle so the two training
  stages share identical feature statistics.

### 1.4 Target parameterization

Rather than predicting absolute gain directly, the model learns the
**residual offset** w.r.t. a physical baseline:

```
baseline_ij = target_gain_i + target_gain_tilt_i * (47 - j) / 94     # j=0..94
target_ij   = (label_ij - baseline_ij) * mask_ij
```

At inference time the offset is added back to the baseline.

### 1.5 Evaluation metrics

All metrics are masked (activated channels only) and implemented in
`[src/ofc_ml/evaluate.py](../src/ofc_ml/evaluate.py)`:

- **MAE / RMSE / MSE** — standard.
- **Kaggle Score** — the competition metric
  ```
  Score = MAE + 0.3 * T95 + 0.1 * Tmax + 0.15 * Std
  T95   = max(0, quantile95(|err|) - MAE - 0.5)
  Tmax  = max(0, max(|err|)         - quantile95(|err|) - 0.7)
  Std   = mean_i std_j(|err_ij|)     over activated j
  ```

We report Overall / Public / Private figures plus per-`Category`
(`aging / shb / unseen / cosmos`) and per-`EDFA_type` (`booster / preamp`)
breakdowns.

### 1.6 Training protocol


| Stage      | Loss       | Optimizer | LR   | Batch | Epochs (max) | Patience | Notes                    |
| ---------- | ---------- | --------- | ---- | ----- | ------------ | -------- | ------------------------ |
| `pretrain` | Masked MSE | Adam      | 1e-3 | 256   | 500          | 40       | COSMOS only              |
| `finetune` | Masked MSE | AdamW     | 2e-4 | 64    | 500          | 60       | Kaggle only              |
| `joint`    | Masked MSE | Adam      | 1e-3 | 256   | 500          | 40       | A-T3, on COSMOS ∪ Kaggle |


All stages use gradient clipping `‖g‖≤1.0`, ReduceLROnPlateau (factor 0.5,
patience 10), and early stopping on masked validation loss (10% of the
respective training set). Seed is fixed at `RANDOM_STATE=42`.

### 1.7 Reference architecture (M-1 "Ours")

`HybridFNOKANPredictor` in `[src/ofc_ml/models/hybrid_fno_kan.py](../src/ofc_ml/models/hybrid_fno_kan.py)`:

```
Input (≈204 dims)
  ↓
[ FourierKAN block ] + residual Linear   ┐
  ↓                                      │ × 5 blocks
Linear head → 95-dim gain-spectrum offset
  ↓
mask ⊙ output
```

Earlier versions of the code included an optional gated FNO-style
`SpectralMixingLayer`, but multi-seed diagnostics showed that the learned gates
remain close to zero and that the extra module does not provide a statistically
stable gain on this dataset.  The paper therefore uses the simpler FourierKAN
backbone as the main architecture and focuses the ablation study on transfer
learning and the physics-grounded target parameterization.

Parameter counts:

| Model                              | role                        | #params |
| ---------------------------------- | --------------------------- | ------- |
| M-0 Wang DNN (Wang et al. 2023)    | external SOTA baseline      | ~132k   |
| M-1 Ours (FourierKAN)              | proposed model              | ~291k   |
| M-2 MLP (same-size)                | architecture ablation       | ~283k   |
| M-3 CNN1D                          | architecture ablation       | ~281k   |
| M-4 Transformer (channel-as-token) | architecture ablation       | ~310k   |

The published Wang DNN is roughly half the size of Ours.  The architecture-
ablation backbones (M-2..M-4) are matched to Ours within ±10 % so that any
gap between them and Ours cannot be attributed to parameter count.


### 1.8 Reproducibility

Each experiment is a YAML under `[experiments/](../experiments)` that
overlays `[experiments/base.yaml](../experiments/base.yaml)`. A single run is

```bash
python scripts/run_experiment.py --config experiments/main/m1_ours.yaml
```

Outputs go to `results/<exp_name>/{metrics.json, submission.csv, model.pt, config.snapshot.yaml}`. Pre-training checkpoints are cached under
`results/_pretrain_cache/` keyed by a hash of (model, pretrain config, data
subsampling, seed) so Main / Ablation experiments that share an identical
pre-training stage do not duplicate work.

---

## 2. Matrix of experiments

The paper claims three contributions: (i) the FourierKAN architecture, (ii)
COSMOS→Kaggle transfer learning, and (iii) physics-baseline target
parameterization.  The experiment matrix is organised so the **headline main
table** isolates the joint impact of all three (Wang DNN vs Ours), while
three independent ablations isolate each contribution.

```mermaid
graph TD
    Root[EDFA Digital Twin Experiments]
    Root --> Main[Main: external SOTA comparison]
    Root --> Abl[Ablations: per-contribution]
    Main --> M0[M-0 Wang DNN, Wang et al. 2023]
    Main --> M1[M-1 Ours FourierKAN]
    Abl --> ARCH[Architecture]
    Abl --> T[Transfer learning]
    Abl --> P[Physics baseline]
    Abl --> D[Data scale]
    ARCH --> A1m[M-1 Ours]
    ARCH --> A2m[M-2 MLP same-size]
    ARCH --> A3m[M-3 CNN1D]
    ARCH --> A4m[M-4 Transformer]
    T --> T1[A-T1 No Finetune]
    T --> T2[A-T2 No Pretrain]
    T --> T3[A-T3 Joint Merge]
    P --> P1[A-P1 Predict Absolute]
    D --> Dp[Pretrain 25 50 100]
    D --> Df[Finetune 25 50 100]
```

### 2.1 Main results (Table 1)

The headline external comparison: our full method versus the published EDFA
DNN baseline by Wang et al. 2023, which is the dataset-origin paper.  Each
row uses its own architecture *and* its own training/transfer protocol, i.e.
this table reflects the joint effect of all three of our contributions.
Source: `[results/_tables/table1_main_summary.csv](../results/_tables/table1_main_summary.csv)`.

- `M-0 Wang DNN` — 4-hidden MLP with BN+ELU+Kaiming init, **Wang's own
  three-phase transfer protocol** (Phase A: head-only retraining, lr=0.05,
  150 epochs; Phase B: full fine-tune with BN frozen, lr=0.001, 20 epochs),
  and direct absolute-gain regression (no physics prior).
- `M-1 Ours` — FourierKAN backbone + physics-baseline residual target +
  Adam pretrain → AdamW finetune.

### 2.2 Architecture ablation (Table 2)

With Ours's transfer protocol and physics baseline held fixed, swap only
the backbone in M-1.  All four backbones are matched to Ours's parameter
count within ±10 %.  Source:
`[results/_tables/table2_architecture_summary.csv](../results/_tables/table2_architecture_summary.csv)`.

- `M-1 Ours` (FourierKAN, reference)
- `M-2 MLP` (same-size LayerNorm + GELU MLP)
- `M-3 CNN1D` (channel-as-sequence convolution)
- `M-4 Transformer` (channel-as-token encoder)

### 2.3 Transfer-learning ablation (Table 3)

Holds backbone (FourierKAN) and physics baseline fixed; varies only the
training protocol.  Source:
`[results/_tables/table3_transfer_summary.csv](../results/_tables/table3_transfer_summary.csv)`.

- `M-1 Ours` — full pretrain → finetune (reference)
- `A-T1 No-Finetune` — use pretrained weights directly on test (zero-shot)
- `A-T2 No-Pretrain` — train from scratch on Kaggle only
- `A-T3 Joint` — single stage on `concat(COSMOS, Kaggle)`

### 2.4 Physics-baseline ablation (Table 4)

Holds backbone (FourierKAN) and protocol (pretrain+finetune) fixed; varies
only the target parameterization.  Source:
`[results/_tables/table4_physics_summary.csv](../results/_tables/table4_physics_summary.csv)`.

- `M-1 Ours` — predict residual offset around the analytical baseline.
- `A-P1 Predict Absolute` — same backbone, regress `calculated_gain_spectra_*`
  directly.

### 2.5 Data-scale ablation (Figure 3)

Two 1-D scans sharing M-1 settings. Source:
`[results/_tables/figure3_data_scale_summary.csv](../results/_tables/figure3_data_scale_summary.csv)`.

- **COSMOS scan** (Kaggle fixed at 100%): `cosmos_ratio ∈ {25%, 50%, 100%}`
- **Kaggle scan** (COSMOS fixed at 100%): `kaggle_ratio ∈ {25%, 50%, 100%}`

Plotted as two line charts (MAE and Kaggle Score vs. ratio).

### 2.6 Note on Wang DNN feature dimension

Wang et al. 2023 trains *one* DNN per EDFA device on a 193-dim feature
vector (`g0`, `P_in`, `P_out`, 95 input-spectrum samples, 95 channel-loading
indicators).  In our setup we instead train a **single global** Wang DNN and
feed it the project-wide preprocessor's ~204-dim feature vector (which adds
`target_gain_tilt` plus a small one-hot of `EDFA_type` and `edfa_index`).
This adapts Wang's per-device design to the unseen-device split present in
the Kaggle test set, while keeping every other architectural and training
detail (hidden widths, BN+ELU, Kaiming init, masked-MSE loss, three-phase
transfer protocol, gradient clipping at 3.0) identical to the published
specification.  The only structural change is the width of the first linear
layer.  Total Wang DNN parameter count under our input is ~132k, in line
with the published ~125k–130k figures.

---

## 3. Reproducing the matrix

### 3.1 End-to-end recipe

```bash
# 1. Set up environment
conda activate ofc_ml
pip install -r requirements.txt

# 2. (Optional, first-time only) convert COSMOS dataset
git clone https://github.com/functions-lab/COSMOS-EDFA-Dataset.git COSMOS-EDFA-Dataset
python scripts/cosmos_to_kaggle.py \
    --cosmos-dataset-dir COSMOS-EDFA-Dataset/dataset \
    --out-dir data/cosmos-as-kaggle \
    --category cosmos --gains 18dB --channel-types fix

# 3. Run the default paper matrix (main + architecture + transfer + physics).
#    `data_scale` runs are excluded by default; use --include-data-scale to add.
python scripts/run_matrix.py

# 4. (Recommended) sweep multiple seeds to populate the cross-seed
#    summary tables that the notebook reads.
python scripts/run_seeds.py --seeds 42 43 44 45 --aggregate-after

# 5. (Or manually) aggregate already-finished runs into paper-ready tables.
python scripts/aggregate_results.py
```

### 3.2 Running subsets

```bash
# Headline external comparison only
python scripts/run_matrix.py --only m0_wang_dnn m1_ours

# Architecture ablation only
python scripts/run_matrix.py --only m1_ours m2_mlp m3_cnn1d m4_transformer

# Transfer-learning ablation only
python scripts/run_matrix.py --only m1_ours a_t1_no_finetune a_t2_no_pretrain a_t3_joint

# Physics-baseline ablation only
python scripts/run_matrix.py --only m1_ours a_p1_predict_absolute

# Include the optional data-scale 1-D scans
python scripts/run_matrix.py --include-data-scale

# Force re-run regardless of existing metrics.json
python scripts/run_matrix.py --force
```

### 3.3 Overriding hyperparameters on the fly

```bash
# Shorten for a smoke test
python scripts/run_matrix.py --override pretrain.epochs=30 finetune.epochs=60

# Run m1_ours on CUDA with a different batch size
python scripts/run_experiment.py --config experiments/main/m1_ours.yaml \
    --override device=cuda:0 pretrain.batch_size=512
```

### 3.4 Recommended compute

- Default matrix = **9 experiments**:
  2 main (M-0 Wang DNN, M-1 Ours)
  + 3 architecture (M-2/M-3/M-4)
  + 3 transfer (A-T1/A-T2/A-T3)
  + 1 physics (A-P1).
  Data-scale adds 6 optional runs (`--include-data-scale`).
- The runner orders experiments so that a single FourierKAN pretrain is
  produced first and reused by the rest of the FourierKAN cells through
  `_pretrain_cache/`; only ~3 unique pretrain runs are actually performed.
- ~45 min/seed on a single A100 (full 500/600 epoch budgets);
  ~5–7 h/seed on MPS (Apple Silicon).

---

## 4. Paper figure inventory


| Figure | Content                                                                                  | Script                                   |
| ------ | ---------------------------------------------------------------------------------------- | ---------------------------------------- |
| Fig. 1 | Predicted vs. ground-truth gain spectrum on three typical samples (aging / shb / unseen) | add `scripts/plot_spectra.py` (TODO)     |
| Fig. 2 | Per-channel error box-plots, four methods side-by-side                                   | add `scripts/plot_channel_err.py` (TODO) |
| Fig. 3 | Data-scale ablation (two line plots)                                                     | from `figure3_data_scale.csv`            |


Fig. 1 / Fig. 2 are intentionally out of scope for the first refactor — they
only consume `results/<exp_name>/submission.csv` plus `test_labels.csv` and
can be added as thin plotting scripts without touching the training code.