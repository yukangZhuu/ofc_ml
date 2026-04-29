# Wang et al. 2023 Section 6: TL-Based EDFA Gain Spectrum Model

本文档总结 Wang et al. 2023 论文中第 6 节 "TL-BASED EDFA GAIN SPECTRUM MODEL" 的 transfer learning 方法与配置。重点包括第 6.A 小节中的模型迁移流程、target dataset 选择，以及第 6.B 到 6.D 小节中讨论的三类迁移场景。

本文档默认标准 DNN architecture 已按论文第 5 节实现，结构为：

```text
193 -> 256 -> 128 -> 128 -> 128 -> 95
```

标准 DNN 架构与非迁移学习训练配置单独整理在 `wang2023_dnn_edfa_gain_model.md`。

## 1. Transfer Learning 的目标

论文提出 transfer learning 的原因是：标准 DNN-based EDFA model 对每个 EDFA 都需要大量 gain spectrum measurements。收集完整数据非常耗时，因此作者希望把一个已经训练好的 source EDFA model 迁移到一个新的 target EDFA、target gain setting 或不同 EDFA type 上。

Transfer learning 的核心目标是：

```text
Use a pre-trained source DNN model
to build a target EDFA gain model
with very few target measurements.
```

论文证明，在若干场景下只使用目标域完整训练集的 `0.5%`，即每个 gain setting 仅 `13` 条 target measurements，就可以得到接近完整 DNN 训练的预测精度。

## 2. Base Model

TL 使用的 source model 是第 5 节训练好的 component-level DNN-based EDFA model。

网络结构不变：

```text
Input layer:    n = 193
Hidden layer 1: n = 256
Hidden layer 2: n = 128
Hidden layer 3: n = 128
Hidden layer 4: n = 128
Output layer:   n = 95
```

整体为：

```text
193 -> 256 -> 128 -> 128 -> 128 -> 95
```

输入仍然是：

```text
[g0, P_in, P_out, S_in(lambda_1), ..., S_in(lambda_95), c_1, ..., c_95]
```

输出仍然是：

```text
[g(lambda_1), ..., g(lambda_95)]
```

Loss 仍然是标准 DNN 中的 masked MSE，只在 loaded channels 上计算：

```text
MSE = 1 / sum_{i=1}^{95} c_i
      * sum_{i: c_i = 1} [g_pred(lambda_i) - g_meas(lambda_i)]^2
```

论文第 6.A 小节文字中说使用 Eq. (4) 的 same MSE loss function。按论文排版，第 5 节中的 MSE 公式编号是 Eq. (5)，这里应理解为继续使用同一个 masked MSE loss。

## 3. Fig. 9 中的 TL 含义

Fig. 9 的图注指出：

```text
Transfer learning reinitializes the orange output layer before retraining.
```

图中橙色层是最终 output layer：

```text
Linear: 128 -> 95
```

这表示 TL 时不会替换整个网络，而是保留 source DNN 的前面部分作为 feature extractor，重新初始化最后的 output layer，然后用少量 target measurements 训练 target model。

可以把 Fig. 9 中的迁移逻辑理解为：

```text
Pre-trained source model:

Input + 4 hidden layers          Output layer
193 -> 256 -> 128 -> 128 -> 128 -> 95
        feature extractor          source head

Transfer learning:

freeze feature extractor
reinitialize output layer
train output layer on target data
then fine-tune all layers
```

## 4. TL Procedure

第 6.A 小节给出了完整的 transfer learning procedure。该流程分为三个阶段。

### 4.1 Stage 1: Freeze Input and Hidden Layers

首先冻结 DNN 的输入层和所有四个 hidden layers。

论文原文含义是：

```text
The input layer and all four hidden layers are frozen.
They are treated as the feature extractor of the DNN model.
```

实现时，除了最后 output layer 以外，所有参数都应设置为不训练：

```python
for param in feature_extractor.parameters():
    param.requires_grad = False
```

Feature extractor 包括：

```text
Input/first dense block
Hidden layer 1
Hidden layer 2
Hidden layer 3
Hidden layer 4
Batch normalization modules in these blocks
```

### 4.2 Stage 2: Reinitialize Output Layer and Retrain

冻结 feature extractor 后，重新初始化 output layer。

Output layer 是最终的：

```text
128 -> 95
```

论文说明 output layer 使用 Kaiming normalization 重新初始化。

然后仅用 target EDFA 的少量 target measurements 重新训练模型。由于 feature extractor 已冻结，训练中实际更新的是 output layer。

该阶段配置为：

```text
Loss:                    masked MSE over loaded channels
Learning rate / step size: 0.05
Epochs:                  150
Trainable part:          reinitialized output layer
Frozen part:             input layer + all four hidden layers
```

实现时可以写成：

```python
reinitialize_output_layer_with_kaiming(model.output_layer)
freeze_feature_extractor(model)
train(model, target_loader, lr=0.05, epochs=150, loss=masked_mse)
```

论文没有在第 6.A 小节提到此阶段是否继续使用 gradient clipping。如果代码实现希望与第 5 节保持训练稳定性，可以沿用标准 DNN 的 gradient clipping threshold `3.0`，但这不是第 6.A 明确重复给出的配置。

### 4.3 Stage 3: Unfreeze All Layers and Fine-Tune

输出层 retraining 之后，论文将所有层 unfreeze，并进行 full-model fine-tuning。

该阶段配置为：

```text
Loss:                    masked MSE over loaded channels
Learning rate / step size: 0.001
Epochs:                  20
Trainable part:          all layers
Special rule:            batch normalization parameters are kept unchanged
```

实现时可以写成：

```python
unfreeze_all_layers(model)
keep_batch_norm_parameters_unchanged(model)
train(model, target_loader, lr=0.001, epochs=20, loss=masked_mse)
```

论文特别强调：

```text
batch normalization parameters are kept unchanged
```

这很重要，因为 target dataset 很小。如果让 BN running statistics 或 BN affine parameters 在极小 target set 上大幅更新，可能会破坏 source model 学到的稳定特征分布。

在 PyTorch 中可以考虑如下处理：

```python
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm1d):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
```

这会保持 BN running mean/variance 不更新，并且冻结 BN 的 affine parameters。论文没有给出框架级细节，但实现时应确保 BN 参数在 fine-tuning 阶段保持不变。

## 5. Source and Target Dataset Size

论文定义：

```text
N_src = number of gain spectrum measurements at each gain setting
        used to train the source model

N_tgt = number of target EDFA measurements at each gain setting
        used to construct the target model through TL
```

标准 source model 使用：

```text
N_src = 2732
```

这是每个 EDFA、每个 gain setting 的标准 DNN training set size。

论文比较了三种 target data size。

第一种：

```text
N_tgt = 5
N_tgt / N_src = 0.2%
Target data type: fully loaded channel configurations
```

第二种：

```text
N_tgt = 13
N_tgt / N_src = 0.5%
Target data type: fully loaded and half-loaded channel configurations
```

第三种：

```text
N_tgt = 41
N_tgt / N_src = 1.5%
Target data type: fully loaded, half-loaded, single-channel-loaded,
                  and double-channel-loaded configurations
```

论文根据 Fig. 14 的结果选择：

```text
N_tgt = 13
N_tgt / N_src = 0.5%
```

原因是 `0.5%` 的 target-source data size ratio 明显优于 `0.2%`，并且与 `1.5%` 的效果相近。因此后续 TL 实验都使用 `N_tgt = 13`。

论文强调，这相当于把 target EDFA 数据采集量减少约：

```text
200x
```

同时在评估场景中仍能达到：

```text
MAE < 0.2 dB across all EDFAs
```

## 6. TL Scenario 1: Between EDFAs of the Same Type

第 6.B 小节研究同类型 EDFA 之间的迁移：

```text
booster -> booster
pre-amplifier -> pre-amplifier
```

设置如下：

```text
Source EDFA: one of eight booster or pre-amplifier EDFAs
Target EDFA: another EDFA of the same type
Gain settings: 15 / 18 / 21 dB
Target data size: N_tgt = 13
N_tgt / N_src = 0.5%
Test sets: random and goalpost
```

论文构造了 MAE matrix。矩阵含义是：

```text
entry (i, i):
  component-level DNN model trained directly on the i-th EDFA
  no transfer learning

entry (i, j), i != j:
  source model is the i-th EDFA
  target model is the j-th EDFA
  obtained through transfer learning
```

观察结果：

- 对每一行，`(i, i)` 通常比 `(i, j)` 更小，因为完整训练的 component-level DNN 使用了大量 target EDFA 数据。
- TL model 使用非常少的 target data，因此误差略大是预期结果。
- TL 仍明显优于传统 CM model，并且数据采集量大幅降低。

同类型迁移的 MAE 范围如下：

```text
Booster EDFAs:
  Random test set:   0.06 - 0.12 dB
  Goalpost test set: 0.08 - 0.18 dB

Pre-amplifier EDFAs:
  Random test set:   0.09 - 0.18 dB
  Goalpost test set: 0.12 - 0.24 dB
```

论文总结：

```text
Target booster model MAE <= 0.18 dB
Target pre-amplifier model MAE <= 0.24 dB
```

整体上，booster 之间迁移比 pre-amplifier 之间迁移效果更好；goalpost channel-loading configurations 比 random configurations 更难预测。

论文还指出，如果在 target data 中加入少量 random/goalpost channel-loading measurements，target model 在这些测试集上的表现可能进一步提升。

## 7. TL Scenario 2: Between Gain Settings of the Same EDFA

第 6.C 小节研究同一个 EDFA 不同 gain settings 之间的迁移。

目标是用一个或两个 source gain settings 训练 source model，然后迁移到新的 target gain setting。

论文中的例子：

```text
15 & 18 -> 21
```

含义是：

```text
Source model is trained using EDFA measurements with 15 dB and 18 dB gain settings.
Then it is transferred to a target model using measurements with 21 dB gain setting.
```

论文比较了两种情况：

```text
one source gain setting -> one target gain setting
two source gain settings -> one target gain setting
```

结果显示，只使用一个 source gain setting 时效果较差：

```text
MAE up to 0.8 dB on random test set
MAE up to 1.0 dB on goalpost test set
```

加入第二个 source gain setting 后，模型具备更多 domain knowledge，误差显著降低。例如加入 `15 dB` 作为第二个 source setting 后：

```text
Random test set MAE:   0.21 dB
Goalpost test set MAE: 0.19 dB
```

论文总结，使用两个 source gain settings 时，所有 source/target gain combinations 上的平均 MAE 为：

```text
0.16 dB
```

论文说明 pre-amplifier EDFAs 中也观察到类似 MAE 表现。

实现启示：

- 如果要把模型迁移到新的 gain setting，最好不要只依赖单个 source gain setting。
- 使用两个 source gain settings 训练 source model 可以显著提升迁移质量。
- Target gain setting 仍只需要少量 target measurements 来完成 output-layer retraining 和 fine-tuning。

## 8. TL Scenario 3: Between EDFA Types

第 6.D 小节研究不同 EDFA type 之间的迁移。

这里的 EDFA type 是：

```text
B = booster
P = pre-amplifier
```

迁移方向包括：

```text
B -> P
P -> B
```

论文是在同一个 ROADM 上做不同类型 EDFA 之间的迁移，并与不使用 TL 的 DNN model 对比：

```text
B -> B: booster DNN without cross-type TL
P -> P: pre-amplifier DNN without cross-type TL
B -> P: source booster model transferred to target pre-amplifier model
P -> B: source pre-amplifier model transferred to target booster model
```

Source model 使用三个 gain settings。

结果：

```text
All target model MAE <= 0.21 dB
Average MAE = 0.16 dB
```

相比 source model，TL 引入的 MAE degradation 为：

```text
Booster / pre-amplifier under random test set:
  0.06 / 0.10 dB

Booster / pre-amplifier under goalpost test set:
  0.10 / 0.13 dB
```

说明即使 source 和 target 是不同 EDFA type，只要在同一个 ROADM 场景下，TL 仍能有效减少 target data collection。

## 9. TL 实现检查清单

实现论文第 6 节 TL 方法时，建议按以下流程：

- 先训练好一个标准 source DNN，结构为 `193 -> 256 -> 128 -> 128 -> 128 -> 95`。
- 使用与标准 DNN 相同的输入、输出和 masked MSE loss。
- 冻结 input layer 和所有四个 hidden layers，把它们作为 feature extractor。
- 将 output layer `128 -> 95` 用 Kaiming normalization 重新初始化。
- 使用少量 target measurements 训练 output layer。
- Output-layer retraining 配置为 step size `0.05`，训练 `150` epochs。
- 然后 unfreeze all layers。
- Fine-tuning 配置为 step size `0.001`，训练 `20` epochs。
- Fine-tuning 阶段保持 batch normalization parameters unchanged。
- 推荐 target dataset size 为 `N_tgt = 13`，即 `N_tgt / N_src = 0.5%`。
- `N_tgt = 13` 应包括 fully loaded 和 half-loaded channel configurations。
- 如果做 gain-setting transfer，优先使用两个 source gain settings，而不是只使用一个。
- 如果做 same-type EDFA transfer，分别处理 booster-to-booster 和 pre-amplifier-to-pre-amplifier。
- 如果做 cross-type EDFA transfer，可以处理 `B -> P` 和 `P -> B`，但应预期误差略高于完整 DNN。

## 10. 和标准 DNN 的主要区别

标准 DNN 训练：

```text
Train all layers from scratch
Use 2732 source/target measurements per gain setting
Learning rate 0.001
Epochs 600
Gradient clipping threshold 3.0
```

TL 训练：

```text
Start from a pre-trained source DNN
Freeze input + hidden layers
Reinitialize output layer
Train output layer with small target set
Unfreeze all layers
Fine-tune briefly while keeping BN parameters unchanged
Use about 13 target measurements per gain setting
```

核心区别是：TL 将 source model 的 input/hidden layers 视为可迁移的 feature extractor，只用很少 target data 学习新的 output mapping，并进行短时间 full-model fine-tuning。

