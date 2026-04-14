# TabPFNv2 架构及使用指南

> University of Michigan | KECC Research Group | Haohong Zhang
>
> Knowledge-Guided TabPFN 研究项目的基础环境与使用文档

---

## 1. 项目简介

TabPFNv2 是一个表格数据的基础模型（Foundation Model），发表于 Nature 2025。它使用 **In-Context Learning (ICL)** 进行预测：将训练集作为"上下文"输入 Transformer，通过一次前向传播直接输出预测结果，**不需要对每个数据集单独训练**。

本仓库记录了：
- TabPFNv2 的环境配置与使用方法(提前说明:这个配置的GPU是密歇根大学的GreatLake数据集,这里可以替换成自己的GPU)
- 从 CSV 表格输入到预测输出的完整流程
- TabPFNv2 的架构分析笔记

---

## 2. 环境配置

### 2.1 Great Lakes 集群（UMich HPC）

```bash
# 登录
ssh <uniqname>@greatlakes.arc-ts.umich.edu

# 首次配置（只做一次）
module load python3.10-anaconda
conda init bash
source ~/.bashrc
conda create -p ~/tabpfn_env python=3.10 -y
conda activate ~/tabpfn_env
pip install tabpfn scikit-learn pandas

# 以后每次登录只需要
conda activate ~/tabpfn_env
```

### 2.2 本地 Mac（用于看代码、小规模测试）

```bash
conda create -n tabpfn python=3.10 -y
conda activate tabpfn
pip install tabpfn scikit-learn pandas
```

### 2.3 验证安装

```bash
python -c "from tabpfn import TabPFNClassifier; print('TabPFN OK')"
```

---

## 3. 使用方法：从 CSV 到预测输出

### 3.1 输入格式

CSV 文件，最后一列为标签（分类目标），其余列为数值特征：

```
age,blood_pressure,cholesterol,blood_sugar,bmi,heart_rate,disease
63,145,233,1,28.5,150,1
37,130,250,0,23.1,187,0
41,130,204,0,26.3,172,0
...
```

### 3.2 完整代码

```python
"""
TabPFNv2 预测脚本
输入: CSV 文件（最后一列为标签）
输出: 每个测试样本的预测标签和概率
"""
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

# ===== 1. 读取数据 =====
df = pd.read_csv("your_data.csv")
print(f"数据形状: {df.shape[0]} 个样本, {df.shape[1]-1} 个特征")

# 前 n-1 列为特征，最后一列为标签
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ===== 2. 分割数据 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")

# ===== 3. TabPFNv2 预测（核心只有3行）=====
model = TabPFNClassifier()        # 加载预训练模型
model.fit(X_train, y_train)       # 存储训练数据（不是训练）
y_pred = model.predict(X_test)    # 一次前向传播，直接出结果
y_proba = model.predict_proba(X_test)

# ===== 4. 输出结果 =====
acc = (y_pred == y_test).mean()
print(f"\n准确率: {acc:.4f}")

print(f"\n{'样本':>4} {'真实':>4} {'预测':>4} {'各类概率'}")
print("-" * 40)
for i in range(len(y_test)):
    probs = " ".join([f"{p:.3f}" for p in y_proba[i]])
    print(f"{i+1:>4} {y_test[i]:>4} {y_pred[i]:>4}   [{probs}]")
```

### 3.3 运行

```bash
# 本地
python run_prediction.py

# Great Lakes 上
conda activate ~/tabpfn_env
python run_prediction.py
```

### 3.4 输出示例

```
数据形状: 40 个样本, 6 个特征
训练集: 32 个样本
测试集: 8 个样本

准确率: 0.8750

样本 真实 预测 各类概率
----------------------------------------
   1    0    0   [0.912 0.088]
   2    1    1   [0.034 0.966]
   3    0    0   [0.887 0.113]
   ...
```

---

## 4. 与 XGBoost 的对比

```python
"""对比 TabPFNv2 和 XGBoost 的准确率与速度"""
import time
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TabPFNv2
start = time.time()
tabpfn = TabPFNClassifier()
tabpfn.fit(X_train, y_train)
pred_tabpfn = tabpfn.predict(X_test)
time_tabpfn = time.time() - start
acc_tabpfn = (pred_tabpfn == y_test).mean()

# XGBoost
start = time.time()
xgb = XGBClassifier(n_estimators=100, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
time_xgb = time.time() - start
acc_xgb = (pred_xgb == y_test).mean()

print(f"TabPFNv2:  准确率={acc_tabpfn:.4f}, 耗时={time_tabpfn:.2f}s")
print(f"XGBoost:   准确率={acc_xgb:.4f}, 耗时={time_xgb:.2f}s")
```

---

## 5. 项目文件结构

```
tabpfn_project/
├── README.md                    # 本文件
├── ARCHITECTURE.md              # TabPFNv2 架构分析
├── sample_medical.csv           # 示例数据
├── run_prediction.py            # 通用预测脚本
├── run_medical.py               # 医疗数据示例
├── test_tabpfn.py               # 基础测试
└── trace_tabpfn.py              # 输入输出追踪
```

---

## 6. 相关论文

| 论文 | 发表 | 核心贡献 |
|------|------|----------|
| TabPFNv2 | Nature 2025 | 首个全面超越树模型的表格基础模型 |
| TARTE | arXiv 2025.07 | 知识预训练，融入字符串语义 |
| TabICL | ICML 2025 | 可扩展到 500K 样本的 ICL |

---

## 7. 参考资源

- TabPFNv2 源码: https://github.com/PriorLabs/TabPFN
- TabICL 源码: https://github.com/soda-inria/tabicl
- TARTE 数据集: https://huggingface.co/datasets/inria-soda/carte-benchmark
