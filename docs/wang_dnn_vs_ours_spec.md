# Wang-DNN-TL Baseline vs. Ours: Implementation Spec

This document fixes the exact configuration used for the **Wang-DNN-TL (global
adapted)** baseline in this project, in direct comparison with our main method
(**M-1 Ours**). It is the implementation reference for the YAML configs under
`experiments/main/` and the model code under `src/ofc_ml/models/`.

The high-level paper-facing rationale is in
[`docs/experiment.md`](experiment.md). The Wang paper distillations live in
[`docs/wang2023_dnn_edfa_gain_model.md`](wang2023_dnn_edfa_gain_model.md) and
[`docs/wang2023_tl_edfa_gain_model.md`](wang2023_tl_edfa_gain_model.md).

---

## 1. Side-by-side summary

| Aspect | Ours (`m1_ours`) | Wang-DNN-TL (global adapted, `wang_dnn_tl`) |
|---|---|---|
| Model registry name | `hybrid_fno_kan` | `wang_dnn` |
| Backbone | FourierKAN blocks + residual projections | Dense MLP with BatchNorm + ELU |
| Hidden widths | `[256, 256, 128, 128, 128]` (5 blocks) | `[256, 128, 128, 128]` (4 blocks, paper Fig. 9) |
| Activation | GELU (inside FourierKAN block) | ELU (after every hidden BN) |
| Normalisation | LayerNorm (inside FourierKAN block) | BatchNorm1d after every hidden Linear |
| Spectral mixing | Optional, disabled by default | Not used |
| Output head | `Linear(prev, 95)`, masked | `Linear(128, 95)`, masked |
| Init | Xavier (linear), N(0, 0.02) (Fourier amps) | Kaiming-normal (all dense layers) |
| Target parameterization | Residual: `label - baseline`, baseline added back at inference | Absolute: predict `calculated_gain_spectra_*` directly |
| Loss | Masked MSE on activated channels | Masked MSE on activated channels |
| Modeling granularity | Global digital twin (one model, all EDFAs) | Global digital twin (adapted; original paper is per-device) |
| Pretrain dataset | COSMOS (subsample by `cosmos_ratio`) | COSMOS (same) |
| Finetune dataset | OFC/Kaggle (subsample by `kaggle_ratio`) | OFC/Kaggle (same) |
| Pretrain optimizer | Adam, lr `1e-3`, weight decay `1e-4` | Adam, lr `1e-3`, weight decay `1e-4` (matches Wang) |
| Finetune optimizer | AdamW, lr `2e-4`, weight decay `5e-5` | AdamW, lr `2e-4`, weight decay `5e-5` (our TL protocol) |
| Pretrain epochs | `500` with early stop (patience `40`) | Same. Wang's standalone DNN budget is `600` epochs; with early stopping on COSMOS the cap is non-binding. |
| Finetune epochs | `500` with early stop (patience `60`) | Same |
| Gradient clipping | `‖g‖ ≤ 1.0` | `‖g‖ ≤ 3.0` (Wang's value) |
| BN handling during finetune | N/A (no BN in FourierKAN) | BN frozen (eval mode + non-trainable affine), see §3.6 |
| Scheduler | ReduceLROnPlateau (factor `0.5`, patience `10`) | Same |
| Validation split | `5%` of training set | Same |
| Inference postprocessing | `pred + baseline`, then mask | `pred`, then mask (no baseline addition because target is absolute) |

---

## 2. Where Wang-DNN-TL deliberately differs from the Wang paper

The original Wang DNN paper (Sec. 5) trains one component-level DNN per EDFA;
the TL paper (Sec. 6) freezes the source feature extractor, reinitialises the
output layer, and fine-tunes on `N_tgt ≈ 13` per-device measurements. We do
**not** reproduce that flow as the main baseline. Concretely:

1. **Modeling granularity**: We train a single model over all EDFAs in the
   benchmark. Per-device modeling is incompatible with the global digital-twin
   task and with our small per-device sample counts in OFC/Kaggle.
2. **TL protocol**: We use our COSMOS pretrain → OFC/Kaggle finetune protocol
   for both methods, instead of Wang's freeze-output-layer-then-unfreeze
   procedure. This isolates the architectural variable (`wang_dnn` vs.
   `hybrid_fno_kan`) plus the target parameterisation (`absolute` vs.
   `residual`) while holding the transfer protocol fixed.
3. **Target dataset size**: Wang's TL recipe is tuned for very small target
   sets (`N_tgt = 5/13/41`); our setting uses the OFC/Kaggle training set as a
   whole. We do not subsample to match Wang's regime.
4. **Input feature schema**: Wang's input vector is
   `[g0, P_in, P_out, S_in, c]` of width 193. Ours additionally one-hot
   encodes `EDFA_type` and `edfa_index` so the global model can disambiguate
   devices. The Wang baseline uses our input schema as-is, accepting the
   widened input dimension; this is a global-adaptation choice, not a faithful
   reproduction.
5. **BatchNorm freezing in fine-tuning**: Wang's TL Stage 3 keeps BN
   parameters fixed. Our fine-tuning fully unfreezes BN because we do not run
   the freeze-output-layer stage at all.

These deviations are paper-fair: every difference is shared with Ours (or
deliberately retained for fidelity to Wang's standalone DNN training where it
does not conflict with the global setting), so the main comparison still
isolates the architecture and target choices.

What we **do** keep faithful to Wang:

- Hidden widths `[256, 128, 128, 128]`.
- ELU activation in hidden blocks.
- BatchNorm1d on each hidden block.
- Linear output `→ 95` channels with no activation.
- Kaiming-normal initialisation for all `Linear` layers.
- Masked MSE training loss.
- Adam optimizer with lr `1e-3` for the COSMOS-pretrain stage.
- Gradient clipping at `3.0` (Wang's standalone DNN value).

---

## 3. Wang-DNN module specification

File: `src/ofc_ml/models/wang_dnn.py`.

The class signature mirrors `MLPPredictor`, so it plugs into the existing
`Trainer` and `predict_test` paths without changes:

```python
class WangDNNPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 95,
        hidden_dims: list[int] | None = None,   # default [256, 128, 128, 128]
        dropout: float = 0.0,
    ): ...

    def forward(self, x, mask=None):
        # body: [Linear -> BatchNorm1d -> ELU (-> Dropout)] * len(hidden_dims)
        # head: Linear(prev, output_dim)
        # if mask is not None: out = out * mask
        ...
```

Initialisation:

- For every `Linear` layer (hidden + output): `kaiming_normal_(weight,
  nonlinearity="relu")` and zero bias. Wang only specifies "Kaiming
  normalization" without a `nonlinearity` argument; `"relu"` is the standard
  Kaiming setting and is the closest standard match for ELU in
  `torch.nn.init`.
- `BatchNorm1d` uses PyTorch defaults.

Dropout is parameterised but **defaults to `0.0`** to match Wang. We expose it
only for ablation experiments that may want to add regularisation.

The `mask` argument follows the same convention as `HybridFNOKANPredictor`: if
provided, the output is element-wise multiplied by it so non-activated
channels output zero.

---

### 3.6 BatchNorm freezing during finetune

Without this rule, the Wang-DNN-TL baseline produces catastrophic test-time
outliers when run end-to-end on the OFC/Kaggle benchmark. We observed:

```text
seed 42, full pretrain + finetune:
  pretrain val loss: 0.003 dB^2  (RMSE ≈ 0.05 dB)
  finetune val loss: 0.008 dB^2  (RMSE ≈ 0.09 dB)
  test overall MAE: 3.92 dB, Tmax: 75.8 dB
  unseen-category MAE: 4.58 dB
```

By contrast `a_p1_predict_absolute` (FourierKAN with `predict_absolute=true`,
no BatchNorm) under the same protocol gives MAE 0.82 dB and Tmax 0.0 dB, and
m1_ours (FourierKAN with the residual target) gives MAE 0.10 dB. The gap
between Wang-DNN-TL and `a_p1_predict_absolute` is therefore *not* explained
by the absolute-vs-residual target choice — it is an additional, BN-specific
failure mode.

Diagnosis:

- COSMOS has 41 440 rows; the BN running statistics learned during pretrain
  are well-conditioned for the global EDFA distribution.
- OFC/Kaggle has 473 rows. With `batch_size=64` and roughly 7 effective
  batches per fine-tune epoch, the BN running statistics get rapidly
  overwritten by a small, biased view of the data.
- At test time the OOD samples (`Category=unseen`, including
  `target_gain=20.0` which never appears in training) are passed through
  these contaminated BN layers and the activations diverge.
- The `predict_absolute=True` target then has no analytical anchor to clamp
  the divergence, so individual channels can produce errors up to ~75 dB.

Fix:

For the entire finetune stage, every `BatchNorm1d` (or any BN-family) module
in the Wang-DNN model is held in `eval()` mode and its `weight` / `bias`
have `requires_grad=False`. This preserves the COSMOS-pretrain BN statistics
through finetune and keeps the affine parameters from overfitting to the
small Kaggle batches.

Implementation:

- A new `StageConfig.freeze_batchnorm` flag (default `False`) is added in
  `src/ofc_ml/configs/schema.py`.
- `Trainer` in `src/ofc_ml/model.py` re-applies the freeze at the start of
  every epoch so that `model.train()` does not undo it.
- `experiments/main/wang_dnn_tl.yaml` sets `finetune.freeze_batchnorm: true`.
  The pretrain stage on COSMOS leaves BN active and trainable, which is the
  paper-faithful Wang standalone-DNN configuration.

This rule matches Wang TL paper Stage 3 (`docs/wang2023_tl_edfa_gain_model.md`
§4.3): *"batch normalization parameters are kept unchanged"*. Even though we
do not adopt Wang's full freeze-output-layer-then-unfreeze TL flow, the
BN-freeze rule is the right thing to do whenever a BN-using model is
fine-tuned on a small target set, irrespective of which TL protocol is used
around it.

The `freeze_batchnorm` field is intentionally excluded from the
`_arch_signature()` cache key whenever it equals its default `False`, so
existing pretrain caches produced before this field was introduced remain
valid for all other experiments.

## 4. Wang-DNN-TL training configuration

File: `experiments/main/wang_dnn_tl.yaml` (see also `experiments/base.yaml`
for the unchanged shared defaults).

Key fields:

```yaml
base: ../base.yaml
name: wang_dnn_tl

model:
  name: wang_dnn
  hidden_dims: [256, 128, 128, 128]
  dropout: 0.0
  predict_absolute: true        # paper-faithful: predict absolute gain spectrum

stages: [pretrain, finetune]

pretrain:
  grad_clip: 3.0                # Wang paper standalone DNN value
  drop_last: true               # see §3.7

finetune:
  grad_clip: 3.0
  drop_last: true               # see §3.7
  freeze_batchnorm: true        # see §3.6
```

Everything not overridden is inherited from `experiments/base.yaml`, which
already encodes Wang's standalone DNN settings on the pretrain side
(`Adam`, lr `1e-3`, weight decay `1e-4`, batch size `256`).

### 3.7 Per-stage `drop_last`

`BatchNorm1d` cannot accept a singleton tail batch in training mode, so the
training DataLoader for any Wang-DNN stage drops its trailing partial batch.
This is exposed as a per-stage flag `StageConfig.drop_last` (default `False`)
and explicitly set to `true` for both Wang stages.

It is **not** turned on globally for non-BN models. We previously enabled
`drop_last=True` unconditionally inside `make_dataloaders` (commit
`d87ba13`), but that systematically dropped the last ~200 COSMOS samples
per pretrain epoch (out of ~39368) and the last 1 Kaggle sample per
finetune epoch. Across hundreds of epochs the deterministic-tail drop
shifted m1_ours and m2_mlp's optimisation trajectories enough to flip
their relative ranking on the OFC test set. Reverting `drop_last` to
False for non-BN models reproduced the pre-`d87ba13` numbers bit-for-bit
(see commit message of the per-stage `drop_last` fix).

---

## 5. Physics baseline specification

The physics baseline is added alongside Wang-DNN-TL in the main table. It is
non-neural and serves as a lower bound for the analytical gain/tilt model.

File: `experiments/main/physics_baseline.yaml`.

Implementation strategy:

- A `ZeroPredictor` model class returns a `(B, 95)` zero tensor.
- The YAML sets `model.name: physics_baseline`,
  `model.predict_absolute: false`, and `stages: []`.
- `orchestrate()` runs no training stage because `stages` is empty.
- `predict_test()` already adds the baseline back when
  `predict_absolute=False`:

  ```text
  out = baseline + zero_pred = baseline
  out = out * mask
  ```

  So the saved submission is exactly `baseline * mask`, matching the
  analytical reference.

This avoids special-casing inside the runner: the same code path produces a
metrics.json and a submission.csv, just with zero training time.

---

## 6. Output and integration

For both new experiments the runner writes the canonical files:

```
results/seed_<S>/wang_dnn_tl/
    metrics.json
    submission.csv
    model.pt
    pretrain.pt
    config.snapshot.yaml
    config.source.yaml
    history.csv
    logs/
results/seed_<S>/physics_baseline/
    metrics.json
    submission.csv
    model.pt              # contains an empty / dummy state_dict
    config.snapshot.yaml
    config.source.yaml
    history.csv           # empty (header only)
    logs/
```

These integrate with:

- `scripts/run_experiment.py` (no changes needed beyond the new model classes
  and YAMLs).
- `scripts/run_matrix.py` (extended `ORDERED_EXPERIMENTS`).
- `scripts/run_seeds.py` (no changes needed).
- `scripts/aggregate_results.py` (extended `MAIN_EXPS` and a new `ARCH_EXPS`
  group; see `docs/experiment.md` §5 for the final paper-facing table layout).
