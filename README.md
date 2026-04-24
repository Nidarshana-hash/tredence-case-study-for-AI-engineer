#  Self-Pruning Neural Network (Learned Gating)

##  Project Overview

This project implements a **dynamic model compression technique** where a neural network learns to prune its own weights during training.

Unlike traditional post-training pruning, this approach uses **learnable gates** and **L1 regularization** to identify and remove redundant connections **on the fly**.

---

##  Key Idea

Each weight is associated with a learnable gate:

[
W_{eff} = W \odot \sigma(\text{gate_scores})
]

* Gates are values between **0 and 1**
* If a gate → **0**, the weight is effectively **pruned**
* The network learns which connections are unnecessary

---

## Core Features

### 🔹 Custom Prunable Linear Layer

* Implements gating mechanism on weights
* Learns both weights and gate parameters

### 🔹 Sparsity-Aware Loss

[
\text{Total Loss} = \mathcal{L}*{task} + \lambda \cdot \mathcal{L}*{sparsity}
]

* **Task Loss** → CrossEntropy
* **Sparsity Loss** → L1 norm of gates
* Encourages many gates → 0

### 🔹 Dynamic Adaptation

* Model structure evolves during training
* Automatically balances **accuracy vs efficiency**

---


## Implementation Details

### 🔸 Gate Transformation

* Sigmoid function converts gate scores → [0,1]

### 🔸 L1 Regularization

* Penalizes non-zero gates
* Forces many gates toward **zero**

### 🔸 Thresholding

* Gates < **0.05** are considered pruned

### 🔸 Result

* Bimodal distribution:

  * Gates ≈ 0 → pruned
  * Gates ≈ 1 → active

---

## Results Visualization

* Histogram of gate values shows:

  * Spike near **0** (pruned weights)
  * Cluster away from 0 (important weights)

---

## Key Insights

* Small λ → High accuracy, low sparsity
* Large λ → High sparsity, lower accuracy
* Trade-off between **model size and performance**

---
