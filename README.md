# Highway Network - PyTorch Implementation

This repository contains a PyTorch implementation of the **Highway Network**, a neural network architecture that enables efficient information flow across layers using *gating mechanisms*.  
It extends the traditional feedforward neural network by introducing *transform* and *carry* gates that dynamically regulate how much of the input is transformed or passed through directly to address vanishing gradient problem.

---

## Architecture Overview

A **Highway Layer** can be expressed as:

\[
y = H(x) * T(x) + x * (1 - T(x))
\]

Where:
- **H(x)** → Transform function (e.g., Linear + ReLU)
- **T(x)** → Transform gate (sigmoid activation)
- **(1 - T(x))** → Carry gate (controls how much input bypasses transformation)

---

| Component          | Description                                |
| ------------------ | ------------------------------------------ |
| **Input Layer**    | Linear transformation of input features    |
| **Highway Layers** | Combination of transform and carry gates   |
| **Output Layer**   | Produces class scores or regression output |
| **Activation**     | ReLU (for H), Sigmoid (for T)              |

---

## Author
Muhammd Ali

---
## Reference
Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015).[read the paper here][https://arxiv.org/abs/1505.00387]
