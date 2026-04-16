# TabPFNv2 完整工作流：输入 → 代码 → 输出

> Haohong Zheng | University of Michigan
>
> 在医疗数据集上运行 TabPFNv2 的完整记录

---

## 1. 输入数据：`sample_medical.csv`

模拟的心脏病预测数据集，40个病人，6个特征：

| 列名 | 含义 | 类型 | 范围 |
|------|------|------|------|
| age | 病人年龄 | 数值 | 37–71 |
| blood_pressure | 血压 | 数值 | 105–172 |
| cholesterol | 胆固醇 | 数值 | 168–417 |
| blood_sugar | 空腹血糖是否 >120 mg/dl | 二元 | 0 或 1 |
| bmi | 体重指数 | 数值 | 22.4–32.1 |
| heart_rate | 最大心率 | 数值 | 113–188 |
| **disease** | **心脏病（预测目标）** | **二元** | **0=无病, 1=有病** |

原始 CSV（前10行）：

```csv
age,blood_pressure,cholesterol,blood_sugar,bmi,heart_rate,disease
63,145,233,1,28.5,150,1
37,130,250,0,23.1,187,0
41,130,204,0,26.3,172,0
56,120,236,0,24.8,178,0
57,140,192,0,27.1,148,1
57,120,354,0,30.2,163,0
56,140,294,0,25.6,153,1
44,120,263,0,22.4,173,0
52,172,199,1,31.5,162,1
57,150,168,0,26.8,174,1
```

完整数据集：共40行（20个无病，20个有病）。

---

## 2. 代码：`run_medical.py`

```python
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("sample_medical.csv")
print(f"数据: {df.shape[0]}个病人, {df.shape[1]-1}个特征")

X = df.iloc[:, :-1].values    # 前6列 = 特征
y = df.iloc[:, -1].values     # 最后一列 = 标签 (disease)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TabPFNClassifier()        # 加载预训练模型
model.fit(X_train, y_train)       # 存储训练数据（不是训练！）
y_pred = model.predict(X_test)    # 一次前向传播 → 预测结果
y_proba = model.predict_proba(X_test)

print(f"\n{'No':>4} {'Real':>4} {'Pred':>4} {'P(0)':>8} {'P(1)':>8}")
print("-" * 36)
for i in range(len(y_test)):
    print(f"{i+1:>4} {y_test[i]:>4} {y_pred[i]:>4} {y_proba[i][0]:>8.3f} {y_proba[i][1]:>8.3f}")

acc = (y_pred == y_test).mean()
print(f"\nAccuracy: {acc:.4f}")
```

### 每行代码的作用：

| 代码 | 作用 |
|------|------|
| `pd.read_csv("sample_medical.csv")` | 读取 CSV 文件 |
| `df.iloc[:, :-1].values` | 取除最后一列外的所有列作为特征（X） |
| `df.iloc[:, -1].values` | 取最后一列作为标签（y） |
| `train_test_split(X, y, test_size=0.2)` | 分成80%训练集（32人）和20%测试集（8人） |
| `TabPFNClassifier()` | 加载预训练好的 TabPFNv2 模型（43MB权重） |
| `model.fit(X_train, y_train)` | 把训练数据存为上下文（这不是训练！） |
| `model.predict(X_test)` | 一次前向传播通过12层 Transformer → 出预测 |
| `model.predict_proba(X_test)` | 同一次前向传播 → 出类别概率 |

---

## 3. 如何运行

### 在 Great Lakes（UMich 集群）上：

```bash
# 登录
ssh ********@greatlakes.arc-ts.umich.edu

# 激活环境
conda activate ~/tabpfn_env
cd ~/tabpfn_project

# 运行
python run_medical.py
```

### 在本地 Mac 上：

```bash
conda activate tabpfn
python run_medical.py
```

---

## 4. 实际输出（来自 Great Lakes）

```
数据: 40个病人, 6个特征

  No Real Pred     P(0)     P(1)
------------------------------------
   1    1    1    0.007    0.993
   2    1    1    0.446    0.554
   3    0    0    0.965    0.035
   4    1    1    0.013    0.987
   5    1    1    0.202    0.798
   6    0    0    0.979    0.021
   7    1    1    0.252    0.748
   8    0    0    0.966    0.034

Accuracy: 1.0000
```

### 输出含义：

| 列 | 含义 |
|----|------|
| No | 测试病人编号（1–8） |
| Real | 真实标签（0=无病, 1=有病） |
| Pred | 模型预测（0 或 1） |
| P(0) | 预测为无病的概率 |
| P(1) | 预测为有病的概率 |

### 逐个分析：

| 病人 | 真实 | 预测 | 无病概率 | 有病概率 | 分析 |
|------|------|------|----------|----------|------|
| 1 | 有病 | 有病 | 0.007 | 0.993 | 模型99.3%确信有病，非常自信 |
| 2 | 有病 | 有病 | 0.446 | 0.554 | 正确但不太确信（55.4%），边界病例 |
| 3 | 无病 | 无病 | 0.965 | 0.035 | 模型96.5%确信无病，非常自信 |
| 4 | 有病 | 有病 | 0.013 | 0.987 | 模型98.7%确信有病，非常自信 |
| 5 | 有病 | 有病 | 0.202 | 0.798 | 正确，79.8%确信 |
| 6 | 无病 | 无病 | 0.979 | 0.021 | 模型97.9%确信无病，非常自信 |
| 7 | 有病 | 有病 | 0.252 | 0.748 | 正确，74.8%确信 |
| 8 | 无病 | 无病 | 0.966 | 0.034 | 模型96.6%确信无病，非常自信 |

**8个病人全部预测正确，准确率100%。没有训练过程，一次前向传播直接出结果。**

---

## 5. 如何换成自己的数据

### 第一步：准备 CSV 文件

格式要求：
- 所有特征列是**数值**
- **最后一列**是分类标签
- 没有缺失值（或提前处理好）

示例：

```csv
特征1,特征2,特征3,...,标签
1.5,200,0.8,...,0
3.2,150,1.1,...,1
...
```

### 第二步：修改代码中的文件名

只需要改**一行**：

```python
# 改之前：
df = pd.read_csv("sample_medical.csv")

# 改之后（换成你的文件名）：
df = pd.read_csv("你的数据.csv")
```

其他代码完全不用改。

### 第三步：上传并运行

```bash
# 从 Mac 上传文件到 Great Lakes：
scp ~/Downloads/你的数据.csv ******@greatlakes.arc-ts.umich.edu:~/tabpfn_project/

# 登录并运行：
ssh *****@greatlakes.arc-ts.umich.edu
conda activate ~/tabpfn_env
cd ~/tabpfn_project
python run_medical.py
```

### 重要限制：

| 限制 | 上限 |
|------|------|
| 最大样本数 | ~10,000（超过30K可能崩溃） |
| 最大特征数 | 500 |
| 最大类别数 | 10 |
| 输入类型 | 仅数值（字符串需提前编码） |
| 缺失值 | 需提前处理 |

---

## 6. 内部发生了什么（流程图）

```
你的 CSV 表格（40个病人 × 6个特征）
    │
    ▼
┌─────────────────────────────────┐
│ train_test_split                │
│ 32个病人 → 训练集                 │
│ 8个病人 → 测试集                  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ model.fit(X_train, y_train)     │
│ 把32个训练病人存为"上下文"          │
│ （没有训练！没有梯度下降！）         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ model.predict(X_test)           │
│                                 │
│ 送入12层 Transformer：           │
│   输入：32个训练 + 8个测试         │
│   每层做三件事：                  │
│     ① Row-wise attention       │
│       （特征之间交互）            │
│       "年龄和血压有什么关系？"     │
│     ② Column-wise attention    │
│       （病人之间交互）            │
│       "这个病人跟其他病人比？"     │
│       （测试病人只看训练病人）      │
│     ③ MLP + LayerNorm          │
│                                 │
│ 一次前向传播 → 完成                │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 每个测试病人的输出：               │
│   P(disease=0), P(disease=1)    │
│   预测标签 = 概率更大的那个         │
└─────────────────────────────────┘
```

**没有梯度下降。没有超参数调优。没有训练循环。只有一次前向传播。**
