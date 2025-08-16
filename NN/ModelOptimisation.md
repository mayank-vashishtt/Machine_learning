# Session 4: Advanced Deep Learning & Model Optimization — Detailed Notes

## 1. Overfitting & Underfitting

### Overfitting

* **Definition:** Model learns training data too well, including noise, but fails to generalize to unseen data.
* **Symptoms:**

  * Low training loss.
  * High validation/test loss.
* **Causes:**

  * Too complex a model (many parameters).
  * Insufficient training data.
  * Over-training without regularization.
  * Lack of noise-robust methods.
* **Remedies:**

  * Add regularization (L1, L2, Dropout).
  * Use early stopping.
  * Gather more training data.
  * Reduce model size/complexity.
  * Use data augmentation.

### Underfitting

* **Definition:** Model is too simple to capture underlying data structure.
* **Symptoms:**

  * High training loss.
  * High validation/test loss.
* **Causes:**

  * Model too small or too shallow.
  * Wrong choice of architecture/features.
  * Insufficient training (too few epochs).
* **Remedies:**

  * Increase model complexity (layers, parameters).
  * Use better feature engineering.
  * Train longer.

**Quick Comparison Table:**

| Aspect     | Overfitting (Bad Generalization)         | Underfitting (Too Simple)          |
| ---------- | ---------------------------------------- | ---------------------------------- |
| Train Loss | Low                                      | High                               |
| Test Loss  | High                                     | High                               |
| Fixes      | Regularization, more data, simpler model | Bigger/better model, more training |

---

## 2. Regularization Techniques

### L1 Regularization (Lasso)

* Adds **absolute penalty** (‖w‖₁) to weights.
* Encourages **sparsity**: some weights become exactly zero.
* Useful for **feature selection**.

**Loss Function:**
L = Original Loss + λ Σ |w|

### L2 Regularization (Ridge)

* Adds **squared penalty** (‖w‖₂²) to weights.
* Shrinks weights smoothly but rarely zeros them out.
* Useful for stabilizing models.

**Loss Function:**
L = Original Loss + λ Σ w²

### Dropout

* Randomly disables neurons during training.
* Prevents reliance on specific neurons.
* Improves robustness and reduces overfitting.

**Comparison Table:**

| Method  | What It Does           | Effect                 | Use Case                       |                             |                        |
| ------- | ---------------------- | ---------------------- | ------------------------------ | --------------------------- | ---------------------- |
| L1      | Penalizes              | w                      |                                | Sparsity, feature selection | When features are many |
| L2      | Penalizes w²           | Shrinks weights        | General-purpose regularization |                             |                        |
| Dropout | Drops neurons randomly | Robustness, redundancy | Deep networks, large datasets  |                             |                        |

---

## 3. Batch Normalization (BatchNorm)

* Normalizes **activations** per mini-batch (mean 0, variance 1).
* Benefits:

  * Stabilizes gradients.
  * Allows higher learning rates.
  * Speeds up convergence.
  * Acts as mild regularizer.
* Placement: Before or after activation depending on architecture.

---

## 4. Advanced Activation Functions

| Function       | Formula / Definition       | Usage & Key Points                                                     |
| -------------- | -------------------------- | ---------------------------------------------------------------------- |
| **ReLU**       | f(x) = max(0, x)           | Default choice; avoids vanishing gradients, fast.                      |
| **Leaky ReLU** | f(x) = max(0.01x, x)       | Fixes “dead neurons” problem.                                          |
| **ELU**        | Smooth curve for negatives | Sometimes better convergence than ReLU.                                |
| **Sigmoid**    | 1 / (1+e⁻ˣ)                | Maps to (0,1); used in binary classification; vanishing gradient risk. |
| **Tanh**       | (eˣ – e⁻ˣ)/(eˣ + e⁻ˣ)      | Maps to (–1,1); better than sigmoid, still vanishing issues.           |
| **Softmax**    | exp(xᵢ)/Σ exp(xⱼ)          | Converts logits → probability distribution in multi-class tasks.       |

---

## 5. Modern Optimizers

### Stochastic Gradient Descent (SGD)

* Updates weights using gradient from a mini-batch.
* Simple, but sensitive to learning rate.

### SGD + Momentum

* Adds a “memory” of past gradients.
* Smooths updates, avoids oscillations.
* Faster convergence on difficult terrains.

### RMSProp

* Adaptive learning rate per parameter.
* Keeps a running average of squared gradients.
* Works well for non-stationary problems, often used in RNNs.

### Adam (Adaptive Moment Estimation)

* Combines Momentum + RMSProp.
* Tracks both mean and variance of gradients.
* **Most widely used optimizer** — robust and fast.

**Optimizer Summary Table:**

| Optimizer    | Flavor                      | When to Use                  |
| ------------ | --------------------------- | ---------------------------- |
| SGD          | Basic gradient descent      | Simple problems, baselines   |
| SGD+Momentum | Momentum added              | Deep/tricky nets             |
| RMSProp      | Per-parameter adaptive rate | RNNs, non-stationary data    |
| Adam         | Momentum + RMSProp          | Default choice in most cases |

---

## 6. Key Takeaways

* **Overfitting:** Low train loss, high test loss. Fix with regularization, more data, simpler model.
* **Underfitting:** High train & test loss. Fix with more complex models or longer training.
* **Regularization:**

  * L1 = sparsity, feature selection.
  * L2 = shrink weights, stability.
  * Dropout = robustness.
* **BatchNorm:** Stabilizes training, speeds convergence.
* **Activations:** ReLU & variants dominate; sigmoid/softmax used for outputs.
* **Optimizers:** Adam is the robust default, but SGD variants remain useful for fine-tuning.

⚡ These optimization strategies make deep learning models train faster, generalize better, and perform reliably in real-world scenarios.
