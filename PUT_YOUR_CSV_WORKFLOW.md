# TabPFNv2 Complete Workflow: Input → Code → Output

> Haohong Zheng | University of Michigan
>
> A step-by-step record of running TabPFNv2 on a medical dataset

---

## 1. Input Data: `sample_medical.csv`

A simulated heart disease prediction dataset with 40 patients and 6 features:

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| age | Patient age | Numerical | 37–71 |
| blood_pressure | Blood pressure | Numerical | 105–172 |
| cholesterol | Cholesterol level | Numerical | 168–417 |
| blood_sugar | Fasting blood sugar > 120 mg/dl | Binary | 0 or 1 |
| bmi | Body mass index | Numerical | 22.4–32.1 |
| heart_rate | Maximum heart rate | Numerical | 113–188 |
| **disease** | **Heart disease (target label)** | **Binary** | **0 = no, 1 = yes** |

Raw CSV (first 10 rows):

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

Full dataset: 40 rows total (20 disease=0, 20 disease=1).

---

## 2. Code: `run_medical.py`

```python
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("sample_medical.csv")
print(f"data: {df.shape[0]}个病人, {df.shape[1]-1}个特征")

X = df.iloc[:, :-1].values    # First 6 columns = features
y = df.iloc[:, -1].values     # Last column = label (disease)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TabPFNClassifier()        # Load pre-trained model
model.fit(X_train, y_train)       # Store training data (NO training happens)
y_pred = model.predict(X_test)    # Single forward pass → predictions
y_proba = model.predict_proba(X_test)

print(f"\n{'No':>4} {'Real':>4} {'Pred':>4} {'P(0)':>8} {'P(1)':>8}")
print("-" * 36)
for i in range(len(y_test)):
    print(f"{i+1:>4} {y_test[i]:>4} {y_pred[i]:>4} {y_proba[i][0]:>8.3f} {y_proba[i][1]:>8.3f}")

acc = (y_pred == y_test).mean()
print(f"\nAccuracy: {acc:.4f}")
```

### What each line does:

| Line | Purpose |
|------|---------|
| `pd.read_csv("sample_medical.csv")` | Read the CSV file into a DataFrame |
| `df.iloc[:, :-1].values` | Extract all columns except the last one as features (X) |
| `df.iloc[:, -1].values` | Extract the last column as labels (y) |
| `train_test_split(X, y, test_size=0.2)` | Split into 80% training (32 patients) and 20% test (8 patients) |
| `TabPFNClassifier()` | Load the pre-trained TabPFNv2 model (43MB weights) |
| `model.fit(X_train, y_train)` | Store training data as context (this is NOT training) |
| `model.predict(X_test)` | One forward pass through the 12-layer Transformer → predictions |
| `model.predict_proba(X_test)` | Same forward pass → class probabilities |

---

## 3. How to Run

### On Great Lakes (UMich HPC):

```bash
# Login
ssh ********@greatlakes.arc-ts.umich.edu

# Activate environment
conda activate ~/tabpfn_env
cd ~/tabpfn_project

# Run
python run_medical.py
```

### On local Mac:

```bash
conda activate tabpfn
python run_medical.py
```

---

## 4. Actual Output (from Great Lakes)

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

### Reading the output:

| Column | Meaning |
|--------|---------|
| No | Test patient number (1–8) |
| Real | Ground truth label (0 = no disease, 1 = has disease) |
| Pred | Model prediction (0 or 1) |
| P(0) | Predicted probability of NO disease |
| P(1) | Predicted probability of HAS disease |

### Observations:

- **All 8 patients predicted correctly** (Accuracy = 1.0000)
- Patient 1: Real=1, Pred=1, P(1)=0.993 → Model is 99.3% confident this patient has disease
- Patient 3: Real=0, Pred=0, P(0)=0.965 → Model is 96.5% confident this patient is healthy
- Patient 2: Real=1, Pred=1, P(1)=0.554 → Correct but less confident (55.4%), borderline case
- Patient 6: Real=0, Pred=0, P(0)=0.979 → Model is 97.9% confident this patient is healthy

---

## 5. How to Use Your Own Data

### Step 1: Prepare your CSV

Your CSV must follow this format:
- All feature columns are **numerical**
- The **last column** is the classification label
- No missing values (or handle them beforehand)

Example:

```csv
feature1,feature2,feature3,...,label
1.5,200,0.8,...,0
3.2,150,1.1,...,1
...
```

### Step 2: Replace the filename in the code

Change **one line** in `run_medical.py`:

```python
# BEFORE:
df = pd.read_csv("sample_medical.csv")

# AFTER (replace with your file):
df = pd.read_csv("your_data.csv")
```

That's it. Everything else stays the same.

### Step 3: Upload and run

```bash
# From your Mac, upload the file to Great Lakes:
scp ~/Downloads/your_data.csv ******@greatlakes.arc-ts.umich.edu:~/tabpfn_project/

# Login and run:
ssh ******@greatlakes.arc-ts.umich.edu
conda activate ~/tabpfn_env
cd ~/tabpfn_project
python run_medical.py
```

### Important constraints:

| Constraint | Limit |
|------------|-------|
| Max samples | ~10,000 (above 30K may crash) |
| Max features | 500 |
| Max classes | 10 |
| Input type | Numerical only (encode strings first) |
| Missing values | Handle before input |

---

## 6. What Happens Inside (Summary)

```
Your CSV table (40 patients × 6 features)
    │
    ▼
┌─────────────────────────────────┐
│ train_test_split                 │
│ 32 patients → training set       │
│ 8 patients → test set            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ model.fit(X_train, y_train)     │
│ Store 32 training patients      │
│ as "context" (NO training)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ model.predict(X_test)           │
│                                 │
│ Feed into 12-layer Transformer: │
│   Input: 32 train + 8 test      │
│   Each layer:                   │
│     ① Row-wise attention       │
│       (features interact)       │
│     ② Column-wise attention    │
│       (patients interact)       │
│       (test attends to train)   │
│     ③ MLP + LayerNorm          │
│                                 │
│ Single forward pass → done      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Output for each test patient:   │
│   P(disease=0), P(disease=1)    │
│   Predicted label = argmax      │
└─────────────────────────────────┘
```

**No gradient descent. No hyperparameter tuning. No training loop. Just one forward pass.**
