# TabPFNv2 架构分析

> Haohong Zhang | University of Michigan
>
> 基于源码阅读和论文分析的架构拆解笔记

---

## 1. 总体架构概览

TabPFNv2 是一个 12 层的 Transformer，使用 **二维注意力机制（Two-way Attention）** 处理表格数据。它的核心思想是 **In-Context Learning (ICL)**：将训练集作为上下文输入模型，通过一次前向传播直接预测测试集。

### 1.1 整体数据流

```
输入: CSV 表格
  │
  ▼
┌─────────────────────────────────────┐
│  Preprocessing（预处理）              │
│  缺失值处理、类别编码、标准化          │
│  preprocessing.py                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Encoder（编码器）                    │
│  每个数值 → d维向量                   │
│  encoders.py                         │
│                                      │
│  输入: (batch, n_samples, n_features)│
│  输出: (batch, n_samples, n_features, d_model) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  12 × PerFeatureEncoderLayer        │
│  每层做三件事:                        │
│    ① Row-wise Attention（特征之间）   │
│    ② Column-wise Attention（样本之间）│
│    ③ MLP + LayerNorm                 │
│  layer.py                            │
│                                      │
│  shape 始终: (batch, n_samples, n_features, d_model) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Output Head（输出头）               │
│  向量 → 类别概率                     │
│  transformer.py                      │
└──────────────┬──────────────────────┘
               │
               ▼
输出: 每个测试样本的类别概率
```

### 1.2 源码文件对应关系

```
GitHub: src/tabpfn/architectures/base/

transformer.py       → 总控制器，串联 encoder + layers + output head
layer.py             → 单层实现，包含二维注意力（核心文件）
attention/
  full_attention.py  → 标准 Multi-Head Attention 的计算
encoders/            → 数值到向量的编码
mlp.py               → 前馈网络
config.py            → 模型配置（层数、头数、维度等）
memory.py            → 内存优化
bar_distribution.py  → 回归任务的输出分布
```

---

## 2. 核心：二维注意力机制

### 2.1 什么是二维注意力

表格数据有两个维度：**行（样本）** 和 **列（特征）**。普通 Transformer 只在一个维度上做 Attention。TabPFNv2 在两个维度交替做：

```
一张表格:
         特征1   特征2   特征3
样本1  [  a      b      c  ]    ← Row-wise: a, b, c 之间做 attention
样本2  [  d      e      f  ]
样本3  [  g      h      i  ]
         ↑
         Column-wise: a, d, g 之间做 attention
```

- **Row-wise Attention（特征之间）**：同一个样本内，不同特征互相交流。例如"年龄"和"收入"之间的关系。
- **Column-wise Attention（样本之间）**：同一个特征下，不同样本互相交流。例如不同病人的"血压"相互比较。

### 2.2 为什么用 transpose 实现

标准 Attention 默认沿**倒数第二维**做。TabPFNv2 通过 `transpose(1, 2)` 切换方向：

```python
# 数据 shape: (batch, n_samples, n_features, d_model)

# Row-wise: 不需要 transpose
# attention 沿 n_features 维做（倒数第二维）
# → 同一行的不同特征互相看
output = self.self_attn_between_features(x)

# Column-wise: 需要 transpose
x = x.transpose(1, 2)  # → (batch, n_features, n_samples, d_model)
# attention 沿 n_samples 维做（现在是倒数第二维）
# → 同一列的不同样本互相看
output = self.self_attn_between_items(x)
output = output.transpose(1, 2)  # → 转回原始 shape
```

**同一个 Attention 模块，只是数据摆放方向不同。这就是"二维"的全部实现。**

### 2.3 源码位置

文件：`src/tabpfn/architectures/base/layer.py`

类：`PerFeatureEncoderLayer`

关键代码在 `forward` 方法中（约第230行）：

```python
def forward(self, state, single_eval_pos, ...):
    # state shape: (batch, n_samples, n_features, d_model)

    # 构建子层列表
    sublayers = []

    # Step 1: Row-wise Attention（特征之间）
    sublayers.append(attn_between_features)

    # Step 2: Column-wise Attention（样本之间）
    sublayers.append(attn_between_items)

    # Step 3: MLP
    sublayers.append(self.mlp)

    # 依次执行每个子层，每步后跟 LayerNorm
    for sublayer, layer_norm in zip(sublayers, self.layer_norms):
        state = sublayer(state)
        state = layer_norm(state)

    return state
```

---

## 3. 单层的完整流程

```
输入 state: (batch, n_samples, n_features, d_model)
  │
  ▼
┌─────────────────────────────────────────┐
│ ① attn_between_features (Row-wise)      │
│    不需要 transpose                      │
│    同一行的特征互相做 attention            │
│    "年龄和血压有什么关系？"               │
│    → LayerNorm                           │
├─────────────────────────────────────────┤
│ ② attn_between_items (Column-wise)      │
│    transpose(1,2) 交换样本维和特征维      │
│    同一列的样本互相做 attention            │
│    "这个人的血压跟其他人比怎么样？"        │
│    做完后 transpose(1,2) 转回来           │
│    → LayerNorm                           │
├─────────────────────────────────────────┤
│ ③ MLP (前馈网络)                         │
│    非线性变换                             │
│    → LayerNorm                           │
└─────────────────────────────────────────┘
  │
  ▼
输出 state: (batch, n_samples, n_features, d_model)
  │
  ▼
（重复 12 次）
```

---

## 4. Attention 的底层实现

文件：`src/tabpfn/architectures/base/attention/full_attention.py`

类：`MultiHeadAttention`

这是标准的 Multi-Head Attention，被 layer.py 调用。核心计算：

```python
def forward(self, x, x_kv=None, ...):
    # 1. 计算 Q, K, V
    q, k, v = self.compute_qkv(x, x_kv, ...)
    # Q: "我想找什么信息？"
    # K: "我有什么信息？"
    # V: "我的具体内容是什么？"

    # 2. 做 attention
    # attention_score = softmax(Q @ K^T / sqrt(d_k))
    # output = attention_score @ V
    attention_output = compute_attention_heads(q, k, v, ...)

    # 3. 输出映射
    output = einsum("... h d, h d s -> ... s", attention_output, w_out)
    return output
```

### 4.1 ICL 的关键：测试集只看训练集

在 `attn_between_items`（Column-wise Attention）中，有一个关键设计：

```python
# single_eval_pos: 训练集的结束位置

# 测试样本做 attention 时：
x_query = x[:, single_eval_pos:]   # 测试集的 query
x_kv = x[:, :single_eval_pos]      # 训练集作为 key 和 value

# 测试样本只能 attend 到训练样本
# 训练样本之间也互相 attend
# 这就是 ICL 的核心：测试集从训练集中"学习"
```

这确保了信息从训练集流向测试集，而不会有数据泄露。

---

## 5. 模型配置

```
层数:           12
注意力头数:      暂未确认（需在 config.py 中查看）
隐藏维度:        d_model（由 config.emsize 决定）
前馈网络维度:    dim_feedforward
预训练数据:      合成数据（通过 SCM/DAG 生成）
预训练样本数:    约 1.3 亿个合成数据集
最大支持样本:    ~10K（30K 以上可能内存溢出）
最大支持类别:    10
```

---

## 6. 与 TabICL 的架构对比

| 维度 | TabPFNv2 | TabICL |
|------|----------|--------|
| 注意力方式 | 交替 column-wise 和 row-wise | 先 column-then-row 压缩，再 ICL |
| 列维度是否折叠 | 不折叠，始终保持 | 先折叠为固定维度向量 |
| 复杂度 | O(m²n + n²m) | O(m²n + n²) |
| 列的"身份"识别 | 随机 feature identifier | Distribution-aware ISAB |
| Collapse 解决方案 | 随机向量 | RoPE |
| 标签融合时机 | Early（一开始就拼接） | Late（仅在 ICL 阶段） |
| 最大支持样本 | ~10K | 500K |

### 6.1 关键区别图示

```
TabPFNv2 每层:
  (batch, n_samples, n_features, d_model)
    → row-wise attention    ← 沿 n_features
    → column-wise attention ← 沿 n_samples（需 transpose）
    → MLP
  shape 始终不变，列维度一直保留

TabICL 两阶段:
  Stage 1: TF_col + TF_row
    (batch, n_samples, n_features) → (batch, n_samples, 512)
    列维度被折叠了！

  Stage 2: TF_icl
    (batch, n_samples, 512) → 预测
    只在样本维度上做 attention，不再有列维度
```

TabPFNv2 的列维度始终存在，所以大数据集时 n²m 很贵。TabICL 先压缩掉列维度，后续 ICL 只有 n²，所以更快。

---

## 7. 与 TARTE 的架构对比

| 维度 | TabPFNv2 | TARTE |
|------|----------|-------|
| 预训练数据 | 合成数据 | 知识库（Wikidata + YAGO） |
| 输入类型 | 纯数值 | 数值 + 字符串 + 日期 |
| 下游使用 | ICL（一次前向传播） | frozen/微调/boosting |
| 列编码 | 随机向量 | 列名的语义嵌入（FastText） |
| 是否 ICL 框架 | 是 | 否（需要额外训练下游模型） |

TARTE 有知识但不在 ICL 框架内，只能通过 boosting 与 TabPFNv2 外部组合（TARTE-B-TabPFNv2）。

---

## 8. 对 Knowledge-Guided TabPFN 的启示

### 8.1 核心 Gap

```
TabPFNv2/TabICL: 有 ICL 能力，但不懂语义（合成数据预训练）
TARTE:           懂语义（知识库预训练），但不在 ICL 框架内

→ 能否在 ICL 框架内部原生地融入语义知识？
```

### 8.2 可能的切入点

1. **在 TabICL 的 row embedding 阶段融入 TARTE 的知识嵌入**
   - TabICL 的 late label fusion 设计提供了天然接口
   - TARTE 嵌入（768维）+ TabICL 嵌入（512维）→ 融合后送入 TF_icl

2. **用知识库约束合成数据生成**
   - 让合成数据的变量间关系更接近真实世界

3. **在 column-wise embedding 中加入列名语义**
   - 在 TabICL 的 TF_col 输出基础上融合列名向量

### 8.3 修改的位置

```
如果要在 TabPFNv2 中融入知识:
  → 修改 layer.py 的 PerFeatureEncoderLayer
  → 在 attn_between_items 之前或之后加入知识嵌入

如果要在 TabICL 中融入知识:
  → 修改 TF_row 的输出，拼接 TARTE 嵌入
  → 或修改 TF_icl 的输入
```

---

## 附录：如何查看模型结构

```python
from tabpfn import TabPFNClassifier

model = TabPFNClassifier()
print(model)  # 打印完整模型结构

# 或者查看参数数量
total_params = sum(p.numel() for p in model.model_.parameters())
print(f"总参数量: {total_params:,}")
```
