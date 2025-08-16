# Session 1: Complete Neural Network Notes (Expanded)

## 1. What is a Neural Network?

A **Neural Network** is a computational model inspired by the human brain. It is built from interconnected layers of nodes, called "neurons," that process information. Neural networks are powerful tools for:

* Classification (e.g., spam vs. not spam)
* Regression (e.g., predicting house prices)
* Pattern recognition
* Natural language tasks (translation, sentiment analysis)
* Image and speech recognition

---

## 2. Structure: Layers, Neurons, Weights, Bias

* **Layers:** Organized into three types:

  * **Input Layer:** Receives raw data.
  * **Hidden Layers:** One or more layers that transform data through weighted connections.
  * **Output Layer:** Produces the final prediction.

* **Neuron:** The basic unit. Each neuron:

  1. Receives input values.
  2. Multiplies them by weights.
  3. Adds a bias.
  4. Applies an activation function.

* **Weights:** Parameters that control the strength/importance of input signals.

* **Bias:** Allows neurons to activate even when inputs are zero; adds flexibility.

---

## 3. Activation Functions (Non-Linearity)

Activation functions introduce non-linearity, allowing neural networks to model complex patterns.

* **ReLU (Rectified Linear Unit):**
  $f(x) = \max(0, x)$
  Fast and widely used in deep networks.

* **Sigmoid:**
  $f(x) = \frac{1}{1 + e^{-x}}$
  Maps input to (0,1), often used for probabilities.

* **Tanh:**
  $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
  Maps input to (-1,1), often used in hidden layers.

👉 Without activation functions, a neural network can only represent linear relationships.

---

## 4. Forward Propagation

The process of moving data through the network:

1. Input values are multiplied by weights.
2. A bias is added.
3. The result passes through an activation function.
4. The output is fed into the next layer.
5. This continues until the final prediction is produced.

---

## 5. Loss Functions (Measuring Error)

Loss functions quantify the difference between predictions and actual values.

* **Mean Squared Error (MSE):**
  $L = \frac{1}{n} \sum (y_{true} - y_{pred})^2$
  Used in regression.

* **Mean Absolute Error (MAE):**
  $L = \frac{1}{n} \sum |y_{true} - y_{pred}|$
  Less sensitive to outliers than MSE.

* **Binary Cross-Entropy:**
  $L = -[y \log(p) + (1-y)\log(1-p)]$
  For binary classification.

* **Categorical Cross-Entropy:**
  $L = -\sum y_i \log(p_i)$
  For multi-class classification.

* **Hinge Loss:**
  $L = \max(0, 1 - y \cdot y_{pred})$
  Used in SVM-like classification.

* **Huber Loss:**
  Hybrid of MSE and MAE; robust to outliers.

⚡ Importance: The loss function guides the learning process.

---

## 6. Backpropagation (Learning from Mistakes)

The algorithm that updates weights and biases:

1. Compute the loss.
2. Calculate the gradient (derivative) of the loss with respect to each weight and bias.
3. Update parameters in the opposite direction of the gradient (gradient descent).
4. Use a learning rate to control step size.

---

## 7. Simple Code Example

```python
x = 2           # input data
target = 5      # expected value
w = 0.5         # weight
b = 0.0         # bias
lr = 0.1        # learning rate

def relu(z): return max(0, z)

for epoch in range(5):
    # Forward pass
    y_pred = relu(x * w + b)
    # Loss (MSE)
    loss = (y_pred - target) ** 2

    # Backpropagation
    if x * w + b > 0:
        grad = 2 * (y_pred - target)
        dw = grad * x
        db = grad * 1
    else:
        dw = 0
        db = 0

    # Update
    w -= lr * dw
    b -= lr * db

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
```

---

## 8. Questions to Practice/Expect

* Define: neuron, weight, bias, activation function, loss function.
* Calculate the output for a neuron with given input, weights, bias, and activation.
* Differentiate between MSE, MAE, and Cross-Entropy losses.
* Why is bias important?
* Why are activation functions necessary?
* Explain forward vs. backpropagation.

---

## 9. Optimization

Optimization is the process of adjusting weights and biases to minimize the loss function.

* **Gradient Descent:** Iteratively adjusts weights opposite to the gradient.
* **Variants:**

  * **Stochastic Gradient Descent (SGD):** Updates using one sample at a time.
  * **Mini-batch Gradient Descent:** Uses a subset of data for updates.
  * **Momentum:** Adds inertia to updates, helps escape local minima.
  * **RMSProp:** Adjusts learning rates based on recent gradient magnitudes.
  * **Adam:** Combines Momentum + RMSProp; most popular optimizer.

⚡ Goal: Find parameters that minimize error and improve predictions.
