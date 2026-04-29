# Experiment Design for the EDFA Digital Twin Paper

This document defines the paper-facing experiment matrix for the EDFA gain-spectrum
prediction project. It is intended as guidance for writing the Experiments section
of the paper and for keeping the implementation matrix aligned with the paper
narrative.

The central claim of this work is that a global EDFA digital twin can be improved
by combining three design choices:

1. A FourierKAN-based neural architecture.
2. A physics-informed residual target around a gain/tilt analytical baseline.
3. A transfer-learning protocol from the large COSMOS-EDFA dataset to the small
   OFC/Kaggle training set.

The experiment matrix should therefore separate two questions:

- Main comparison: Does the full proposed method outperform a representative
  prior EDFA DNN baseline under the current global modeling task?
- Ablations: Which parts of the proposed method are responsible for the gain?

## 1. Task and Evaluation Setting

The task is to predict the 95-channel EDFA gain spectrum for each measurement.
All neural models are evaluated on the same held-out OFC/Kaggle test set using
the same activated-channel mask.

The paper should emphasize that our target setting is a global EDFA digital twin:
one model handles multiple EDFAs, EDFA types, device indices, channel loading
patterns, and operating states. This differs from the component-level setting in
Wang et al. 2023, where one DNN is trained independently for each EDFA device.

All reported metrics should use the same offline evaluator:

- MAE / RMSE / MSE over activated channels.
- Kaggle Score, including the tail-error and per-row standard deviation terms.
- Public / Private split when relevant.
- Optional breakdowns by `Category` and `EDFA_type` for analysis, not necessarily
  for the main paper table.

## 2. Main Experiment

The main table should compare the proposed full method against a small number of
meaningful baselines. It should not include every architecture variant, because
those variants are better interpreted as architecture ablations.

Recommended main table:

| Method | Purpose | Training protocol | Target parameterization |
|---|---|---|---|
| Physics baseline | Non-neural analytical reference | No training | `target_gain + target_gain_tilt` line |
| Wang-DNN-TL (global adapted) | Prior EDFA DNN architecture adapted to our benchmark | COSMOS pretrain -> OFC/Kaggle finetune | Absolute gain |
| Ours | Full proposed method | COSMOS pretrain -> OFC/Kaggle finetune | Physics residual |

The expected main claim is:

> Compared with both the analytical physics baseline and a Wang-style DNN baseline
> trained under the same transfer-learning protocol, the proposed FourierKAN
> residual model achieves lower gain-spectrum prediction error on the global
> OFC/Kaggle EDFA benchmark.

## 3. Wang DNN Baseline

### 3.1 Why This Baseline Is Included

Wang et al. 2023 introduced an open EDFA gain-spectrum dataset and demonstrated
that a dense DNN can outperform conventional physics-based EDFA modeling. Since
that dataset family is closely related to the data used in this project, a
Wang-style DNN is the most relevant literature baseline for the main comparison.

This baseline is stronger and more paper-relevant than a generic MLP baseline:
it uses an architecture and modeling setup explicitly proposed for EDFA gain
spectrum prediction.

### 3.2 What We Reuse from Wang et al.

The Wang-style DNN baseline should preserve the core architectural and training
choices documented from the paper:

- Fully connected feed-forward DNN.
- Layer widths: `193 -> 256 -> 128 -> 128 -> 128 -> 95` in the original
  component-level setting.
- Batch normalization and ELU activation in hidden blocks.
- Linear output layer predicting the 95-channel gain spectrum.
- Kaiming-family initialization for dense layers.
- Masked MSE over activated channels.

In our implementation, the exact input dimension may differ from 193 because the
model is adapted to the current global feature schema. The architecture should
preserve the hidden-layer pattern and Wang-style dense block design rather than
the exact original input dimensionality.

### 3.3 Global Adaptation

The original Wang DNN is a component-level model:

```text
one EDFA device -> one independently trained DNN
```

Our benchmark instead asks for a global digital twin:

```text
many EDFA devices -> one unified model
```

Therefore, the main-paper Wang baseline should be implemented as a global-adapted
Wang DNN. It should use the same train/test split, same global feature schema,
and same evaluation protocol as Ours.

This choice is preferred over strict per-device reproduction for the main paper
because:

- It compares methods under the actual task studied in this work.
- It avoids changing the problem from global modeling to many small per-device
  training problems.
- The OFC/Kaggle fine-tuning set is very small after splitting by device; several
  devices have too few rows for a meaningful standalone model.
- It makes the comparison isolate architecture and target design rather than
  mixing in a different modeling granularity.

Recommended paper wording:

> We implement a Wang-style DNN baseline following the dense architecture used
> for EDFA gain-spectrum modeling in Wang et al., including ELU activations,
> batch normalization, Kaiming initialization, a 95-channel linear gain output,
> and masked-MSE training over active channels. Because our benchmark studies a
> unified multi-EDFA digital twin rather than separate component-level models,
> we adapt this DNN to the same global input feature schema and train/test
> protocol used by all compared methods.

Avoid claiming that this is a strict reproduction of the original per-device
experiments. Use names such as:

- `Wang-DNN-TL`
- `Wang-style DNN`
- `Global Wang-DNN`
- `Wang-DNN adapted to global EDFA modeling`

### 3.4 Transfer-Learning Choice

The main Wang baseline should use the same high-level transfer protocol as Ours:

```text
COSMOS pretrain -> OFC/Kaggle finetune
```

This makes it a strong baseline. It prevents the main comparison from being
explained away by saying that Ours wins only because it uses transfer learning.

We do not include a Kaggle-only Wang DNN in the main table because it is less
informative for the paper's primary claim. It would mostly show that training a
dense DNN from scratch on the small OFC/Kaggle training set is weak, which is
already covered by the transfer-learning ablation.

We also do not reproduce the full Wang transfer-learning procedure in the main
matrix. The original TL setup freezes the feature extractor, reinitializes the
output layer, uses extremely small target sets, and is designed for per-device
component-level transfer. That protocol answers a different question from our
global COSMOS-to-OFC transfer setting. It can be discussed as related work or
left as future work / appendix material, but it should not be the primary
baseline unless the project later adds a dedicated per-device reproduction study.

Recommended paper wording:

> For a fair comparison under our data regime, the Wang-style DNN is trained
> with the same COSMOS pretraining and OFC/Kaggle fine-tuning schedule as the
> proposed model. We intentionally do not use the few-shot per-device TL protocol
> from Wang et al., because that protocol targets component-level model transfer,
> whereas our benchmark evaluates a single global EDFA digital twin.

One implementation detail of Wang's TL protocol that we do retain is the
BatchNorm-freezing rule from Stage 3 of the original procedure. During the
fine-tuning stage we keep every BatchNorm module of the Wang-style DNN in
evaluation mode and freeze its affine parameters, so that the BN statistics
learned on the large COSMOS pretraining set are preserved across the small
OFC/Kaggle fine-tuning set. Without this rule the baseline produces extreme
test-time outliers on the out-of-distribution `unseen` category. The detailed
diagnosis is in [`docs/wang_dnn_vs_ours_spec.md`](wang_dnn_vs_ours_spec.md)
§3.6.

### 3.5 Target Parameterization

The Wang-DNN-TL baseline should predict absolute gain, matching the Wang paper's
standard DNN formulation. It should not use our physics residual target in the
main table, because the residual target is one of our contributions.

This creates a clean main comparison:

```text
Wang-DNN-TL = prior dense EDFA DNN + our global transfer setting
Ours        = FourierKAN + physics residual + global transfer setting
```

The residual target should be isolated separately in the physics ablation.

## 4. Ablation Experiments

The ablations should explain the three proposed design choices. They should be
presented after the main table.

### 4.1 Architecture Ablation

Purpose:

> Test whether FourierKAN is beneficial when the transfer protocol and
> physics-informed residual target are held fixed.

Recommended methods:

| Method | Change relative to Ours |
|---|---|
| Ours | FourierKAN + residual target + pretrain -> finetune |
| Same-capacity MLP | Replace FourierKAN blocks with MLP blocks |
| 1D CNN | Replace global dense/FourierKAN backbone with channel-as-sequence CNN |
| Transformer | Replace backbone with channel-as-token Transformer |

This group corresponds to the previous "main experiment" architecture comparison.
In the paper, it should be explicitly framed as an architecture ablation, not as
the main baseline comparison.

The Wang-DNN-TL baseline does not need to be repeated in this ablation table if
it already appears in the main table.

### 4.2 Physics-Prior Ablation

Purpose:

> Test whether predicting the residual around an analytical gain/tilt baseline is
> better than directly predicting the absolute gain spectrum.

Recommended methods:

| Method | Target |
|---|---|
| Ours | `label - baseline`, then add baseline back at inference |
| Predict absolute | Directly predict `calculated_gain_spectra_*` |

All other settings should match Ours:

- FourierKAN backbone.
- COSMOS pretrain -> OFC/Kaggle finetune.
- Same optimizer, epochs, masking, and evaluation.

This is the cleanest experiment for the physics-informed target contribution.

### 4.3 Transfer-Learning Ablation

Purpose:

> Test whether the two-stage COSMOS pretrain -> OFC/Kaggle finetune protocol is
> necessary for the low-resource OFC/Kaggle setting.

Recommended methods:

| Method | Training protocol |
|---|---|
| Ours | COSMOS pretrain -> OFC/Kaggle finetune |
| No finetune | COSMOS pretrain only, evaluated zero-shot |
| No pretrain | OFC/Kaggle training from scratch |
| Joint training | Single-stage training on COSMOS union OFC/Kaggle |

This ablation replaces the need for a Kaggle-only Wang DNN in the main table:
the `No pretrain` experiment directly tests whether small-data training without
COSMOS pretraining is sufficient.

## 5. Final Paper-Facing Matrix

The recommended paper matrix is:

| Group | Method / Experiment | Main question |
|---|---|---|
| Main comparison | Physics baseline | How strong is the analytical gain/tilt reference? |
| Main comparison | Wang-DNN-TL, global adapted | How strong is a literature EDFA DNN baseline under our global TL setting? |
| Main comparison | Ours | Does the complete proposed method improve over the baselines? |
| Architecture ablation | MLP / CNN1D / Transformer variants | Is FourierKAN the best backbone under the same protocol? |
| Physics ablation | Predict absolute gain | Does the residual target help? |
| Transfer ablation | No finetune / no pretrain / joint | Does two-stage transfer learning help? |

Data-scale scans are not part of the current paper-facing matrix.

## 6. Suggested Interpretation Structure

When writing the Experiments section, use the following logic:

1. Establish the benchmark: global multi-EDFA gain-spectrum prediction with
   masked-channel evaluation.
2. Introduce Wang-DNN-TL as the literature baseline and explain why it is adapted
   to global modeling.
3. Present the main table: physics baseline, Wang-DNN-TL, Ours.
4. Use architecture ablation to attribute gains to FourierKAN.
5. Use physics ablation to attribute gains to residual target modeling.
6. Use transfer ablation to attribute gains to COSMOS pretraining and fine-tuning.

The key distinction to maintain is:

- Main table: compares complete methods and representative baselines.
- Ablation tables: isolate individual design choices inside our method.

