# Session 2: Convolutional Neural Networks (CNNs) & Computer Vision

## 1. What is Computer Vision?

**Computer Vision (CV)** is a field of artificial intelligence that focuses on enabling computers to understand, analyze, and interpret visual data (images, videos). Applications range from facial recognition and medical imaging to self-driving cars.

---

## 2. Why Use Neural Networks for Images?

* Images are **high-dimensional**: A single 28×28 grayscale image has 784 pixels; a color image of 256×256 has nearly 200,000 features.
* Hand-crafting features is complex and often inaccurate.
* **Neural Networks (especially CNNs)** can automatically learn hierarchical patterns directly from raw pixels:

  * Early layers learn simple edges/curves.
  * Middle layers learn textures/shapes.
  * Deeper layers learn complex objects.

👉 This makes CNNs far more efficient than fully connected networks for image tasks.

---

## 3. What is a CNN?

A **Convolutional Neural Network (CNN)** is a specialized neural network architecture designed for visual data.

* Uses **convolutions** with filters to capture spatial/local relationships.
* Reduces the number of parameters compared to fully connected layers.
* Learns both **low-level features** (edges, corners) and **high-level features** (faces, objects).

---

## 4. Key CNN Concepts

* **Convolution:**

  * Operation where a small **filter/kernel** slides across the image.
  * At each step, values are multiplied and summed.
  * Produces a **feature map** that shows where patterns exist.

* **Filters/Kernels:**

  * Small weight matrices (e.g., 3×3, 5×5).
  * Each filter specializes in detecting a certain feature (e.g., vertical edges, circles).

* **Feature Maps:**

  * Outputs of convolutions.
  * Highlight specific features learned by filters.

---

## 5. Pooling Layers

Pooling reduces dimensionality while retaining important features.

* **Max Pooling:**

  * Takes the maximum value from a small region (e.g., 2×2).
  * Keeps strongest features and reduces size.

* **Average Pooling:**

  * Takes the average value of a region.
  * Used less in modern CNNs.

👉 Benefits: Reduces computation, prevents overfitting, provides translation invariance.

---

## 6. Activation Layers

* Introduce **non-linearity** after convolutions.
* Most common: **ReLU (Rectified Linear Unit)**.

  * Converts negative values to 0.
  * Speeds up training and avoids vanishing gradients.

---

## 7. Simple CNN Structure & Example Code

Example: A CNN for MNIST digit recognition.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Explanation:

* **Conv2D:** Applies filters to extract features.
* **MaxPooling2D:** Reduces dimensionality.
* **Flatten:** Converts 2D maps into a vector.
* **Dense Layers:** Fully connected layers for decision-making.
* **Softmax:** Produces probability distribution over 10 classes.

---

## 8. Typical Computer Vision Tasks

* **Image Classification:** Assigns a label to an image (e.g., dog, cat).
* **Object Detection:** Identifies objects and their locations (bounding boxes).
* **Image Segmentation:** Labels each pixel.

  * **Semantic Segmentation:** Groups similar objects.
  * **Instance Segmentation:** Differentiates between individual objects.
* **Other Applications:**

  * Facial recognition
  * Optical Character Recognition (OCR)
  * Pose estimation (human skeleton mapping)
  * Medical imaging (tumor detection, X-ray analysis)

---

⚡ CNNs have revolutionized Computer Vision by automatically learning features, improving accuracy, and enabling cutting-edge applications in AI-driven vision systems.
