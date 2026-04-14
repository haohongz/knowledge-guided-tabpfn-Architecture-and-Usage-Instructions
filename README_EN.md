# TabPFNv2: Architecture and Usage Instructions

> University of Michigan | KECC Research Group | Haohong Zhang
>
> Foundational documentation for the Knowledge-Guided TabPFN research project

---

## 1. Overview

TabPFNv2 is a tabular foundation model published in Nature 2025. It uses **In-Context Learning (ICL)** for prediction: the training set is fed as "context" into a Transformer, and predictions are made in a single forward pass — **no per-dataset training required**.

This repository contains:
- Environment setup and usage instructions for TabPFNv2
- End-to-end pipeline from CSV input to prediction output
- Detailed architecture analysis of TabPFNv2

---

## 2. Environment Setup

### 2.1 Great Lakes Cluster (UMich HPC)

```bash
# Login
ssh <uniqname>@greatlakes.arc-ts.umich.edu

# First-time setup (one-time only)
module load python3.10-anaconda
conda init bash
source ~/.bashrc
conda create -p ~/tabpfn_env python=3.10 -y
conda activate ~/tabpfn_env
pip install tabpfn scikit-learn pandas

# Every subsequent login
conda activate ~/tabpfn_env
```

### 2.2 Local Mac (for code reading and small-scale tests)

```bash
conda create -n tabpfn python=3.10 -y
conda activate tabpfn
pip install tabpfn scikit-learn pandas
```

### 2.3 Verify Installation

```bash
python -c "from tabpfn import TabPFNClassifier; print('TabPFN OK')"
```

---

## 3. Usage: From CSV to Prediction

### 3.1 Input Format

A CSV file where the last column is the label (classification target) and all other columns are numerical features:

```
age,blood_pressure,cholesterol,blood_sugar,bmi,heart_rate,disease
63,145,233,1,28.5,150,1
37,130,250,0,23.1,187,0
41,130,204,0,26.3,172,0
...
```

### 3.2 Complete Code

```python
"""
TabPFNv2 Prediction Script
Input: CSV file (last column = label)
Output: Predicted labels and probabilities for each test sample
"""
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

# ===== 1. Load Data =====
df = pd.read_csv("your_data.csv")
print(f"Shape: {df.shape[0]} samples, {df.shape[1]-1} features")

# First n-1 columns = features, last column = label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ===== 2. Split Data =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ===== 3. TabPFNv2 Prediction (only 3 lines) =====
model = TabPFNClassifier()        # Load pre-trained model
model.fit(X_train, y_train)       # Store data (not training)
y_pred = model.predict(X_test)    # Single forward pass, instant results
y_proba = model.predict_proba(X_test)

# ===== 4. Output Results =====
acc = (y_pred == y_test).mean()
print(f"\nAccuracy: {acc:.4f}")

print(f"\n{'No':>4} {'Real':>4} {'Pred':>4} {'Probabilities'}")
print("-" * 40)
for i in range(len(y_test)):
    probs = " ".join([f"{p:.3f}" for p in y_proba[i]])
    print(f"{i+1:>4} {y_test[i]:>4} {y_pred[i]:>4}   [{probs}]")
```

### 3.3 Run

```bash
# Local
python run_prediction.py

# On Great Lakes
conda activate ~/tabpfn_env
python run_prediction.py
```

### 3.4 Example Output

```
Shape: 40 samples, 6 features
Training set: 32 samples
Test set: 8 samples

Accuracy: 1.0000

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
```

---

## 4. Comparison with XGBoost

```python
"""Compare TabPFNv2 vs XGBoost in accuracy and speed"""
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

print(f"TabPFNv2:  Accuracy={acc_tabpfn:.4f}, Time={time_tabpfn:.2f}s")
print(f"XGBoost:   Accuracy={acc_xgb:.4f}, Time={time_xgb:.2f}s")
```

---

## 5. Repository Structure

```
knowledge-guided-tabpfn-Architecture-and-Usage-Instructions/
├── README.md                    # This file (usage guide)
├── ARCHITECTURE.md              # TabPFNv2 architecture analysis
├── sample_medical.csv           # Example dataset
```

---

## 6. Related Papers

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| TabPFNv2 | Nature 2025 | First tabular foundation model to systematically outperform tree-based models |
| TARTE | arXiv 2025.07 | Knowledge pre-training with string semantics from knowledge bases |
| TabICL | ICML 2025 | Scalable ICL for tables up to 500K samples with distribution-aware embedding |

---

## 7. References

- TabPFNv2 source code: https://github.com/PriorLabs/TabPFN
- TabICL source code: https://github.com/soda-inria/tabicl
- TARTE benchmark datasets: https://huggingface.co/datasets/inria-soda/carte-benchmark
