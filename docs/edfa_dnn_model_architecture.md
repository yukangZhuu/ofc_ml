# DNN-Based EDFA Gain Spectrum Model

This document summarizes Section 5.A, "Architecture of the DNN-Based EDFA Model", from Wang et al. (2023), including the network shown in Fig. 9. It also includes the training configuration from the same section and the transfer-learning procedure described in Section 6.A.

The goal is to provide an implementation-ready specification for a coding agent.

## Task Definition

The model predicts the wavelength-dependent gain spectrum of an erbium-doped fiber amplifier (EDFA).

For each EDFA measurement, the model receives operating-condition features, the input power spectrum, and the channel-loading state. It outputs the predicted EDFA gain for each of the 95 wavelength channels.

The model is component-level: one DNN model is trained for each EDFA device.

## Channel Indexing

The dataset uses 95 wavelength channels on a 50 GHz ITU DWDM grid.

- Channel count: `95`
- First wavelength channel: `lambda_1 = 1529.16 nm`, corresponding to `196.050 THz`
- Last wavelength channel: `lambda_95 = 1566.72 nm`, corresponding to `191.350 THz`

For each channel `i`, where `i = 1, 2, ..., 95`, the measured EDFA gain is:

```text
g(lambda_i) = S_out(lambda_i) - S_in(lambda_i)
```

where:

- `S_in(lambda_i)` is the EDFA input power spectrum at channel `i`.
- `S_out(lambda_i)` is the EDFA output power spectrum at channel `i`.
- The paper uses dB-domain powers, so gain is computed by subtraction.

## Input Features

The DNN input is a single flattened feature vector with dimension `193`.

It is formed by concatenating:

```text
[g0, P_in, P_out, S_in(lambda_1), ..., S_in(lambda_95), c_1, ..., c_95]
```

Feature groups:

- `g0`: EDFA target gain setting.
- `P_in`: total EDFA input power.
- `P_out`: total EDFA output power.
- `S_in(lambda_i)`: EDFA input power at wavelength channel `i`; there are 95 values.
- `c_i`: binary channel-loading indicator for wavelength channel `i`; there are 95 values.

The input dimension is therefore:

```text
1 target gain
+ 1 total input power
+ 1 total output power
+ 95 input channel powers
+ 95 channel-loading indicators
= 193 input features
```

The binary channel-loading vector is:

```text
c = [c_i] for i = 1, 2, ..., 95
```

with:

```text
c_i = 1, if the i-th wavelength channel is switched on
c_i = 0, otherwise
```

Fig. 9 labels the input side as:

```text
Gain setting
Total input power
Total output power
ch1 input power
...
ch95 input power
ch1 status (on/off)
...
ch95 status (on/off)
```

## Output Target

The output layer predicts the EDFA gain spectrum:

```text
[g(lambda_1), g(lambda_2), ..., g(lambda_95)]
```

The output dimension is `95`.

Fig. 9 labels the output side as:

```text
ch1 gain
...
ch95 gain
```

Each output neuron corresponds to the predicted gain of one wavelength channel.

## DNN Architecture

The network shown in Fig. 9 is a fully connected feed-forward DNN.

Layer dimensions:

```text
Input layer:    n = 193
Hidden layer 1: n = 256
Hidden layer 2: n = 128
Hidden layer 3: n = 128
Hidden layer 4: n = 128
Output layer:   n = 95
```

Overall architecture:

```text
193 -> 256 -> 128 -> 128 -> 128 -> 95
```

The paper states that the model consists of:

- one input layer,
- four hidden layers with `256 / 128 / 128 / 128` neurons,
- one output layer.

The neurons are initialized by Kaiming normalization.

For the input and hidden layers, the paper applies:

- batch normalization,
- exponential linear unit activation, abbreviated as ELU.

The output layer is a regression layer and should remain linear. Do not apply ELU to the final output if the goal is to reproduce the paper's gain-spectrum regression model.

## Suggested PyTorch Module Layout

An implementation matching the paper can be structured as:

```python
Linear(193, 256)
BatchNorm1d(256)
ELU()

Linear(256, 128)
BatchNorm1d(128)
ELU()

Linear(128, 128)
BatchNorm1d(128)
ELU()

Linear(128, 128)
BatchNorm1d(128)
ELU()

Linear(128, 95)
```

The paper says batch normalization is applied to the input and hidden layers. Fig. 9 itself only shows the dense-layer topology, not the exact placement of batch-normalization blocks. The practical implementation above applies batch normalization after each hidden linear layer and before ELU, which is the usual interpretation for this architecture.

If strict input-layer batch normalization is desired, an additional `BatchNorm1d(193)` can be applied to the raw input feature vector before the first `Linear(193, 256)`. The paper text does not provide enough detail to determine whether this input normalization is a separate explicit module or part of the first hidden block.

## Initialization

All linear-layer neurons should be initialized using Kaiming normalization.

For PyTorch, this means applying Kaiming initialization to each `nn.Linear` weight, for example:

```python
torch.nn.init.kaiming_normal_(linear.weight)
```

Bias initialization is not specified in the paper. A standard implementation can initialize biases to zero.

## Loss Function

The model is trained using mean squared error between predicted and measured gain spectra, but only across loaded channels.

The paper's loss is:

```text
MSE = 1 / sum_{i=1}^{95} c_i * sum_{i: c_i = 1} [g_pred(lambda_i) - g_meas(lambda_i)]^2
```

where:

- `g_pred(lambda_i)` is the predicted gain in wavelength channel `i`.
- `g_meas(lambda_i)` is the measured gain in wavelength channel `i`.
- `c_i` is the channel-loading indicator.
- `sum_i c_i` is the number of loaded channels in the sample.

Implementation detail:

```python
squared_error = (pred_gain - measured_gain) ** 2
masked_squared_error = squared_error * channel_status
per_sample_loss = masked_squared_error.sum(dim=1) / channel_status.sum(dim=1)
loss = per_sample_loss.mean()
```

Expected tensor shapes:

```text
pred_gain:       [batch_size, 95]
measured_gain:   [batch_size, 95]
channel_status:  [batch_size, 95]
```

Only channels with `channel_status == 1` contribute to the loss.

## Standard DNN Training Configuration

For the component-level DNN model, the paper uses:

```text
Gradient clipping threshold: 3.0
Learning rate:               0.001
Training epochs:             600
```

The paper does not specify the optimizer in Section 5.A. If the surrounding codebase has an established optimizer, use that. Otherwise, Adam is a reasonable default for this kind of DNN regression model, while keeping the paper's learning rate and gradient-clipping threshold.

Gradient clipping should be applied during training with threshold `3.0`.

## Dataset Used for DNN Training and Testing

The paper trains and tests the DNN model using measurements under three gain settings:

```text
15 dB
18 dB
21 dB
```

Although the pre-amplifier EDFAs also have `24 dB` and `27 dB` gain-setting data, the paper excludes those two settings from this DNN training/evaluation setup so that booster and pre-amplifier models use consistently sized datasets.

For each EDFA at each gain setting:

```text
Total measurements: 3168
Training set:       2732 measurements, 86%
Test set:            436 measurements, 14%
```

The test set is composed of:

```text
Fixed Goalpost test set: 216 measurements, 7%
Random Baseline test set: 220 measurements, 7%
```

The test set intentionally includes diverse channel-loading configurations:

- randomly selected active channels,
- groups of close-by channels.

## Transfer Learning Context from Fig. 9

Fig. 9 colors the output layer orange and notes:

```text
Transfer learning reinitializes the orange output layer before retraining.
```

This means that, for transfer learning, the already trained DNN is treated as a feature extractor, and the output layer is replaced or reinitialized before retraining on the target EDFA or target setting.

The base architecture remains:

```text
193 -> 256 -> 128 -> 128 -> 128 -> 95
```

but the training procedure changes.

## Transfer Learning Procedure

Section 6.A describes the transfer-learning procedure for converting a pre-trained source EDFA model into a target EDFA model.

The procedure has three stages.

### Stage 1: Freeze Feature Extractor

Freeze:

- the input layer,
- all four hidden layers.

These frozen layers are treated as the feature extractor of the DNN model.

In implementation terms, all parameters before the output layer should have:

```python
requires_grad = False
```

The output layer remains trainable.

### Stage 2: Reinitialize Output Layer and Retrain It

Reinitialize the output layer using Kaiming normalization.

The output layer is the final mapping:

```text
128 -> 95
```

After reinitializing the output layer, retrain the model using the same masked MSE loss as the standard DNN model.

Training configuration for this stage:

```text
Learning rate / step size: 0.05
Epochs:                  150
Loss:                    masked MSE over loaded channels
```

Only the output layer is trained in this stage because the previous layers are frozen.

### Stage 3: Unfreeze and Fine-Tune All Layers

After output-layer retraining, unfreeze all layers and fine-tune the full model.

Training configuration for fine-tuning:

```text
Learning rate / step size: 0.001
Epochs:                   20
Loss:                     masked MSE over loaded channels
```

Batch-normalization parameters are kept unchanged during this fine-tuning stage.

Implementation detail: keeping batch-normalization parameters unchanged can mean:

- do not update BN affine parameters, if present,
- do not update BN running statistics,
- keep BN modules in evaluation mode during transfer fine-tuning.

The exact framework behavior should be chosen carefully so that batch-normalization statistics from the source model are not overwritten by the very small target dataset.

## Transfer Learning Target Dataset Size

The paper evaluates how much target EDFA data is needed for transfer learning.

Let:

- `N_src = 2732`: number of gain-spectrum measurements at each gain setting used to train the source model.
- `N_tgt`: number of target EDFA measurements used to construct the target model through transfer learning.

The paper considers three target dataset sizes:

```text
N_tgt = 5
```

Fully loaded channel configurations only.

```text
N_tgt = 13
```

Fully loaded and half-loaded channel configurations.

```text
N_tgt = 41
```

Fully loaded, half-loaded, single-channel-loaded, and double-channel-loaded configurations.

The paper empirically selects:

```text
N_tgt = 13
N_tgt / N_src = 0.5%
```

This reduces the amount of target data by about `200x` while still achieving less than `0.2 dB` MAE across all EDFAs in the evaluated setting.

## Implementation Checklist

- Build one model per EDFA device for component-level DNN training.
- Use input dimension `193`.
- Use output dimension `95`.
- Use fully connected dimensions `193 -> 256 -> 128 -> 128 -> 128 -> 95`.
- Use ELU activations in the hidden blocks.
- Use batch normalization in the input/hidden part of the model.
- Use a linear final output layer.
- Initialize linear weights using Kaiming normalization.
- Compute loss only over loaded channels using the binary channel-status vector.
- Train standard DNN models for `600` epochs with learning rate `0.001`.
- Apply gradient clipping with threshold `3.0`.
- For transfer learning, freeze all layers except the output layer, reinitialize the output layer, train it for `150` epochs with step size `0.05`, then unfreeze and fine-tune all layers for `20` epochs with step size `0.001`.
- During transfer fine-tuning, keep batch-normalization parameters unchanged.
