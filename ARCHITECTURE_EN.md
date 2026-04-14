# TabPFNv2 Architecture Analysis

> Haohong Zhang | University of Michigan
>
> Architecture walkthrough based on source code reading and paper analysis

---

## 1. High-Level Architecture

TabPFNv2 is a **12-layer Transformer** that uses a **Two-way Attention** mechanism to process tabular data. Its core idea is **In-Context Learning (ICL)**: the training set is fed as context into the model, and predictions for the test set are produced in a single forward pass — no gradient updates, no hyperparameter tuning.

### 1.1 End-to-End Data Flow

```
Input: CSV table (n samples × m features)
  │
  ▼
┌──────────────────────────────────────────┐
│  Preprocessing                            │
│  Handle missing values, encode categories │
│  Source: preprocessing.py                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Encoder                                  │
│  Map each scalar → d-dimensional vector   │
│  Source: encoders.py                      │
│                                           │
│  Input:  (batch, n_samples, n_features)   │
│  Output: (batch, n_samples, n_features, d_model) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  12 × PerFeatureEncoderLayer              │
│  Each layer performs three operations:     │
│    ① Row-wise Attention (between features)│
│    ② Column-wise Attention (between items)│
│    ③ MLP + LayerNorm                      │
│  Source: layer.py                         │
│                                           │
│  Shape throughout: (batch, n_samples, n_features, d_model) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Output Head                              │
│  Vector → class probabilities             │
│  Source: transformer.py                   │
└──────────────┬───────────────────────────┘
               │
               ▼
Output: Class probabilities for each test sample
```

### 1.2 Source Code Map

```
GitHub: src/tabpfn/architectures/base/

transformer.py        → Main controller: chains encoder + layers + output head
layer.py              → Single layer implementation with two-way attention (KEY FILE)
attention/
  full_attention.py   → Standard Multi-Head Attention computation
encoders/             → Scalar-to-vector encoding
mlp.py                → Feed-forward network
config.py             → Model configuration (num layers, heads, dimensions)
memory.py             → Memory optimization utilities
bar_distribution.py   → Output distribution for regression tasks
```

---

## 2. Core Mechanism: Two-Way Attention

### 2.1 What Is Two-Way Attention?

Tabular data has two dimensions: **rows (samples)** and **columns (features)**. A standard Transformer only applies attention along one dimension. TabPFNv2 alternates between both:

```
A table:
              Feature1  Feature2  Feature3
  Sample1  [    a        b        c    ]  ← Row-wise: a, b, c attend to each other
  Sample2  [    d        e        f    ]
  Sample3  [    g        h        i    ]
              ↑
              Column-wise: a, d, g attend to each other
```

- **Row-wise Attention (between features)**: Within the same sample, different features interact. For example, "age" attends to "income" to discover their relationship.
- **Column-wise Attention (between items/samples)**: Within the same feature, different samples interact. For example, different patients' "blood pressure" values are compared.

### 2.2 Implementation via Transpose

Standard attention operates along the **second-to-last dimension** by default. TabPFNv2 switches direction by using `transpose(1, 2)`:

```python
# Data shape: (batch, n_samples, n_features, d_model)

# Row-wise: No transpose needed
# Attention operates along n_features (second-to-last dim)
# → Different features within the same row attend to each other
output = self.self_attn_between_features(x)

# Column-wise: Transpose required
x = x.transpose(1, 2)  # → (batch, n_features, n_samples, d_model)
# Now n_samples is the second-to-last dim
# → Different samples within the same column attend to each other
output = self.self_attn_between_items(x)
output = output.transpose(1, 2)  # → Transpose back to original shape
```

**The same attention module is used both times — only the data orientation differs. This is the entire implementation of "two-way" attention.**

### 2.3 Source Code Location

- **File**: `src/tabpfn/architectures/base/layer.py`
- **Class**: `PerFeatureEncoderLayer`
- **Method**: `forward` (around line 230)

Key code in the `forward` method:

```python
def forward(self, state, single_eval_pos, ...):
    # state shape: (batch, n_samples, n_features, d_model)

    # Build the list of sublayers
    sublayers = []

    # Step 1: Row-wise Attention (between features)
    sublayers.append(attn_between_features)

    # Step 2: Column-wise Attention (between samples)
    sublayers.append(attn_between_items)

    # Step 3: MLP
    sublayers.append(self.mlp)

    # Execute each sublayer followed by LayerNorm
    for sublayer, layer_norm in zip(sublayers, self.layer_norms):
        state = sublayer(state)
        state = layer_norm(state)

    return state
```

---

## 3. Single Layer: Step-by-Step

```
Input state: (batch, n_samples, n_features, d_model)
  │
  ▼
┌─────────────────────────────────────────────┐
│ ① attn_between_features (Row-wise)          │
│    No transpose needed                       │
│    Features within the same row attend       │
│    to each other                             │
│    "What is the relationship between age     │
│     and blood pressure?"                     │
│    → LayerNorm                               │
├─────────────────────────────────────────────┤
│ ② attn_between_items (Column-wise)          │
│    transpose(1,2) swaps sample and feature   │
│    dimensions                                │
│    Samples within the same column attend     │
│    to each other                             │
│    "How does this patient's blood pressure   │
│     compare to other patients'?"             │
│    transpose(1,2) back after attention       │
│    → LayerNorm                               │
├─────────────────────────────────────────────┤
│ ③ MLP (Feed-Forward Network)                │
│    Non-linear transformation                 │
│    → LayerNorm                               │
└─────────────────────────────────────────────┘
  │
  ▼
Output state: (batch, n_samples, n_features, d_model)
  │
  ▼
(Repeated 12 times)
```

---

## 4. Attention: Low-Level Implementation

- **File**: `src/tabpfn/architectures/base/attention/full_attention.py`
- **Class**: `MultiHeadAttention`

This is a standard Multi-Head Attention module, called by `layer.py`. Core computation:

```python
def forward(self, x, x_kv=None, ...):
    # 1. Compute Q, K, V
    q, k, v = self.compute_qkv(x, x_kv, ...)
    # Q: "What information am I looking for?"
    # K: "What information do I have?"
    # V: "What is my actual content?"

    # 2. Compute attention
    # attention_score = softmax(Q @ K^T / sqrt(d_k))
    # output = attention_score @ V
    attention_output = compute_attention_heads(q, k, v, ...)

    # 3. Output projection
    output = einsum("... h d, h d s -> ... s", attention_output, w_out)
    return output
```

### 4.1 ICL Key Design: Test Set Only Attends to Training Set

In `attn_between_items` (Column-wise Attention), there is a critical design for ICL:

```python
# single_eval_pos: the boundary between training and test samples

# When test samples compute attention:
x_query = x[:, single_eval_pos:]    # Test set queries
x_kv = x[:, :single_eval_pos]       # Training set as keys and values

# Test samples can ONLY attend to training samples
# Training samples attend to each other
# This is the core of ICL: the test set "learns" from the training set
```

This ensures information flows from the training set to the test set without data leakage.

---

## 5. Model Configuration

```
Number of layers:       12
Attention heads:        To be confirmed (defined in config.py)
Hidden dimension:       d_model (set by config.emsize)
Feed-forward dimension: dim_feedforward
Pre-training data:      Synthetic datasets (generated via SCM/DAG)
Pre-training scale:     ~130 million synthetic datasets
Max supported samples:  ~10K (memory overflow above ~30K)
Max supported classes:  10
```

---

## 6. Comparison with TabICL

| Aspect | TabPFNv2 | TabICL |
|--------|----------|--------|
| Attention mechanism | Alternating column-wise & row-wise | Column-then-row compression, then ICL |
| Column dimension | Never collapsed, always maintained | Collapsed into fixed-dim vector first |
| Complexity | O(m²n + n²m) | O(m²n + n²) |
| Column identification | Random feature identifier vectors | Distribution-aware ISAB |
| Collapse mitigation | Random vectors | RoPE (Rotary Positional Embedding) |
| Label fusion | Early (concatenated from the start) | Late (only in ICL stage) |
| Max samples | ~10K | 500K |

### 6.1 Key Architectural Difference

```
TabPFNv2 (each layer):
  (batch, n_samples, n_features, d_model)
    → row-wise attention     ← along n_features
    → column-wise attention  ← along n_samples (requires transpose)
    → MLP
  Shape never changes — column dimension is always present
  → Complexity: O(n²m) per layer, expensive when both n and m are large

TabICL (two stages):
  Stage 1: TF_col + TF_row
    (batch, n_samples, n_features) → (batch, n_samples, 512)
    Column dimension is collapsed!

  Stage 2: TF_icl
    (batch, n_samples, 512) → predictions
    Attention only along sample dimension, no column dimension
  → Complexity: O(n²) for ICL stage, much cheaper
```

---

## 7. Comparison with TARTE

| Aspect | TabPFNv2 | TARTE |
|--------|----------|-------|
| Pre-training data | Synthetic data (SCM/DAG) | Knowledge bases (Wikidata + YAGO) |
| Input types | Numerical only | Numerical + strings + dates |
| Downstream usage | ICL (single forward pass) | Frozen / fine-tuned / boosting |
| Column encoding | Random feature identifiers | Semantic embedding via FastText |
| ICL framework | Yes | No (requires training a downstream model) |

TARTE has world knowledge but operates outside the ICL framework. It can only be combined with TabPFNv2 externally via boosting (TARTE-B-TabPFNv2).

---

## 8. Implications for Knowledge-Guided TabPFN

### 8.1 The Core Gap

```
TabPFNv2 / TabICL:  Have ICL capability, but no semantic knowledge
                     (pre-trained on synthetic data only)

TARTE:              Has semantic knowledge (pre-trained on knowledge bases),
                     but not within the ICL framework

→ Can we natively integrate semantic knowledge INSIDE the ICL framework?
```

### 8.2 Potential Research Directions

**Direction A: Fuse TARTE embeddings into TabICL's row embeddings**
- TabICL's late label fusion design provides a natural interface
- Concatenate TARTE embedding (768-dim) + TabICL embedding (512-dim)
- Feed the fused representation into TF_icl
- Story: "Distribution-aware embedding captures numerical properties; knowledge embedding captures semantic properties. The two are complementary."

**Direction B: Knowledge-guided synthetic data generation**
- Use real-world statistical relationships from knowledge bases to constrain the synthetic data generation process
- Make synthetic data more "real-world-like" without using actual real data

**Direction C: Inject column name semantics into column-wise embedding**
- Add a semantic vector (from FastText/sentence-transformer) to TabICL's TF_col output
- Lightweight alternative to TARTE's full knowledge pre-training

**Direction D: Domain-specialized ICL**
- Fine-tune TabICL's TF_col or TF_row on multiple tables from a specific domain
- Test whether domain specialization improves performance on new tables in the same domain

### 8.3 Where to Modify in the Codebase

```
To integrate knowledge into TabPFNv2:
  → Modify layer.py: PerFeatureEncoderLayer
  → Add knowledge embeddings before or after attn_between_items

To integrate knowledge into TabICL:
  → Modify TF_row output: concatenate TARTE embeddings
  → Or modify TF_icl input to include knowledge representations
```

---

## Appendix: Inspecting Model Structure

```python
from tabpfn import TabPFNClassifier

model = TabPFNClassifier()
print(model)  # Print full model architecture

# Count parameters
total_params = sum(p.numel() for p in model.model_.parameters())
print(f"Total parameters: {total_params:,}")
```
