# Wang et al. 2023 Section 5: DNN-Based EDFA Gain Spectrum Model

本文档总结 Wang et al. 2023 论文中第 5 节 "DNN-BASED EDFA GAIN SPECTRUM MODEL" 的非迁移学习部分，重点整理第 5.A 小节 "Architecture of the DNN-Based EDFA Model" 和 Fig. 9 中展示的神经网络结构、输入输出定义、损失函数、数据划分与训练配置。

本文档只描述标准 DNN-based EDFA gain spectrum model。论文第 6 节的 transfer learning 训练流程单独整理在 `wang2023_tl_edfa_gain_model.md`。

## 1. 任务定义

该模型用于预测 EDFA 的 wavelength-dependent gain spectrum，也就是给定一次 EDFA 测量中的运行状态、输入功率谱和通道加载状态，预测 95 个 wavelength channels 上的 EDFA gain。

模型是 component-level DNN model：

```text
one EDFA device -> one independently trained DNN model
```

也就是说，论文中的标准 DNN 并不是一个统一模型同时预测所有 EDFA，而是每个 EDFA 单独训练一个模型。

## 2. 波长通道定义

数据集覆盖 C-band 中的 95 个 50 GHz ITU DWDM wavelength channels。

通道索引为：

```text
i = 1, 2, ..., 95
```

论文给出的通道范围是：

```text
lambda_1  = 1529.16 nm, corresponding to 196.050 THz
lambda_95 = 1566.72 nm, corresponding to 191.350 THz
```

对第 `i` 个通道，EDFA gain 由输出功率谱减去输入功率谱得到：

```text
g(lambda_i) = S_out(lambda_i) - S_in(lambda_i)
```

这里的功率谱使用 dB-domain 数值，因此 gain 可以通过相减得到。

## 3. 输入特征结构

Fig. 9 左侧展示了 DNN 的输入层。输入是一个长度为 `193` 的扁平向量，由以下特征拼接而成：

```text
[g0, P_in, P_out, S_in(lambda_1), ..., S_in(lambda_95), c_1, ..., c_95]
```

各部分含义如下：

- `g0` 是 EDFA target gain setting，例如 `15 dB`、`18 dB`、`21 dB`。
- `P_in` 是 EDFA total input power。
- `P_out` 是 EDFA total output power。
- `S_in(lambda_i)` 是第 `i` 个 wavelength channel 上的 EDFA input power spectrum，共 95 个值。
- `c_i` 是第 `i` 个 wavelength channel 的 channel-loading indicator，共 95 个值。

输入维度计算为：

```text
1 target gain setting
+ 1 total input power
+ 1 total output power
+ 95 input channel powers
+ 95 channel-loading indicators
= 193 input features
```

Fig. 9 中输入侧的标注可以还原为：

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

## 4. Channel-Loading Indicator

论文定义二值通道加载向量：

```text
c = [c_i]_{i=1}^{95}
```

其中：

```text
c_i = 1, if the i-th wavelength channel is switched on
c_i = 0, otherwise
```

实现时，`c` 既是模型输入的一部分，也是 masked loss 的 mask。它的 shape 通常为：

```text
channel_status: [batch_size, 95]
```

## 5. 输出结构

DNN 输出层预测完整 EDFA gain spectrum：

```text
[g(lambda_1), g(lambda_2), ..., g(lambda_95)]
```

输出维度为 `95`。

Fig. 9 右侧输出层标注为：

```text
ch1 gain
...
ch95 gain
```

每一个输出神经元对应一个 wavelength channel 的 predicted gain。

## 6. Fig. 9 网络结构还原

Fig. 9 展示的是一个 fully connected feed-forward DNN。

层宽度如下：

```text
Input layer:    n = 193
Hidden layer 1: n = 256
Hidden layer 2: n = 128
Hidden layer 3: n = 128
Hidden layer 4: n = 128
Output layer:   n = 95
```

整体结构为：

```text
193 -> 256 -> 128 -> 128 -> 128 -> 95
```

可以用下面的 ASCII 图表示 Fig. 9 的主要数据流：

```text
Input features, n = 193
  [g0, P_in, P_out, S_in(lambda_1..lambda_95), c_1..c_95]
        |
        v
Hidden layer 1, n = 256
        |
        v
Hidden layer 2, n = 128
        |
        v
Hidden layer 3, n = 128
        |
        v
Hidden layer 4, n = 128
        |
        v
Output layer, n = 95
  [g(lambda_1), ..., g(lambda_95)]
```

论文说明 neurons are initialized by Kaiming normalization。实现时应对所有 `Linear` 层权重使用 Kaiming 初始化。

## 7. Batch Normalization 与 Activation

论文写明：

```text
For the input and hidden layers, we apply batch normalization and use the exponential linear unit (ELU) activation function.
```

也就是说，输入/隐藏部分使用：

```text
Batch Normalization
ELU activation
```

输出层是 gain spectrum regression layer，应保持线性输出，不应在最后一层之后使用 ELU、ReLU 或 sigmoid。

一个贴近论文描述的 PyTorch module layout 是：

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

Fig. 9 本身只画出了 dense-layer topology，没有明确画出 batch normalization 的具体插入位置。上面的实现采用常见解释：每个 hidden linear block 后接 batch normalization 和 ELU。

如果需要更严格地对应论文中 "input and hidden layers" 的表述，也可以在原始 `193` 维输入向量进入第一层之前增加一个 `BatchNorm1d(193)`。但论文没有明确说明这个 input-layer batch normalization 是单独模块，还是泛指第一层之后的归一化。

## 8. 权重初始化

论文说明神经元使用 Kaiming normalization 初始化。

PyTorch 中可以对每个 `nn.Linear` 的权重执行：

```python
torch.nn.init.kaiming_normal_(linear.weight)
```

Bias 初始化论文没有说明。实现时可以使用常见默认策略，例如将 bias 初始化为 zero。

如果使用 ELU activation，Kaiming 初始化的 nonlinearity 参数需要谨慎选择。论文只给出 "Kaiming normalization"，没有给出 PyTorch 里的具体参数。为了复现论文意图，核心是所有 dense layer 都不要使用 Xavier/random default，而是显式使用 Kaiming-family initialization。

## 9. 损失函数

论文使用 predicted gain spectrum 与 measured gain spectrum 之间的 mean squared error，但只在 loaded channels 上计算。

损失函数为：

```text
MSE = 1 / sum_{i=1}^{95} c_i
      * sum_{i: c_i = 1} [g_pred(lambda_i) - g_meas(lambda_i)]^2
```

其中：

- `g_pred(lambda_i)` 是模型预测的第 `i` 个 wavelength channel gain。
- `g_meas(lambda_i)` 是实测的第 `i` 个 wavelength channel gain。
- `c_i` 是 channel-loading indicator。
- `sum_i c_i` 是当前样本中 switched-on channels 的数量。

实现时应使用 masked MSE：

```python
squared_error = (pred_gain - measured_gain) ** 2
masked_squared_error = squared_error * channel_status
per_sample_loss = masked_squared_error.sum(dim=1) / channel_status.sum(dim=1)
loss = per_sample_loss.mean()
```

推荐 tensor shape：

```text
pred_gain:       [batch_size, 95]
measured_gain:   [batch_size, 95]
channel_status:  [batch_size, 95]
```

注意，关闭的 channel 不参与 loss。即使输出层会产生 95 个通道的预测值，也只有 `c_i = 1` 的通道贡献训练误差。

## 10. 标准 DNN 训练配置

论文对每个 EDFA 都用相同设置训练 component-level DNN：

```text
Gradient clipping threshold: 3.0
Learning rate:               0.001
Training epochs:             600
```

论文第 5.A 小节没有说明 optimizer 类型。实现时可以沿用当前代码库已有 optimizer；如果没有上下文，Adam 是合理默认选择，但这不是论文明确给出的信息。

训练时应保留 gradient clipping：

```text
clip gradient threshold = 3.0
```

## 11. DNN 训练与测试数据选择

论文使用 `15 / 18 / 21 dB` 三个 gain settings 的 EDFA gain spectrum measurements 来训练和测试 DNN-based EDFA gain model。

虽然 pre-amplifier EDFAs 还有额外的 `24 / 27 dB` gain settings，但论文为了让 booster 和 pre-amplifier 使用一致规模的数据集，只选择 `15 / 18 / 21 dB`。

对每个 EDFA、每个 gain setting：

```text
Total measurements: 3168
Training set:       2732 measurements, 86%
Test set:            436 measurements, 14%
```

测试集由两类 channel-loading configurations 构成：

```text
Fixed Goalpost test set: 216 measurements, 7%
Random Baseline test set: 220 measurements, 7%
```

测试集设计用于覆盖 diverse channel-loading configurations，包括随机选择的通道和相邻/成组通道。

## 12. 实现检查清单

实现标准 DNN-based EDFA model 时，至少应满足：

- 每个 EDFA device 训练一个 component-level DNN。
- 输入维度为 `193`。
- 输出维度为 `95`。
- Dense layer 结构为 `193 -> 256 -> 128 -> 128 -> 128 -> 95`。
- Hidden layers 使用 ELU。
- 输入/隐藏部分使用 batch normalization。
- 最终输出层保持 linear regression output。
- Linear weights 使用 Kaiming normalization 初始化。
- Loss 使用 channel mask，只在 switched-on channels 上计算 MSE。
- 标准训练使用 learning rate `0.001`，训练 `600` epochs。
- 训练中使用 gradient clipping threshold `3.0`。
- 训练/测试主要使用 `15 / 18 / 21 dB` gain settings。

