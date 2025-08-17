
# Session 7: Dataset Augmentation, Generative Models, and Core NN Concepts (Detailed Notes)

## 1) Dataset Augmentation

### 1.1 Purpose

* Goal: Artificially enlarge and diversify the training set to improve generalization and reduce overfitting.
* Idea: Show the model realistic variations it may see at inference time.
* Label-preserving: Most image augmentations keep the label unchanged (e.g., a cat is still a cat after a small rotation).

### 1.2 Common Techniques (Vision)

* Geometric: Horizontal/vertical flip, small rotations (±5–30°), random crop & resize, translate (shift), scale/zoom, shear, perspective.
* Photometric: Color jitter (brightness/contrast/saturation/hue), gamma, grayscale, solarize, posterize, equalize.
* Noise/Blur: Gaussian noise, salt–pepper, Gaussian blur, motion blur.
* Occlusion & Mixing: Cutout/Random Erasing (mask regions), MixUp (convex blend of two images/labels), CutMix (patch-level mix with reweighted labels).
* Policy-Based: AutoAugment, RandAugment, TrivialAugmentWide, AugMix.

Why it works: Forces invariance to small geometric/photometric changes; improves robustness to occlusion and label noise; combats class imbalance when applied selectively.

### 1.3 When & How to Apply

* On-the-fly (online) inside the training dataloader → infinite variety, minimal storage.
* Offline pre-generated augmented files → faster epoch time but large storage.
* Per-class policies: Stronger augments for underrepresented classes.
* Test-Time Augmentation (TTA): Average predictions over several mild transforms at inference.

### 1.4 Caveats

* Don’t overdo transforms that change semantics (e.g., heavy rotation on digits 6 vs 9).
* Keep transforms task-consistent (e.g., vertical flip often invalid for medical X-rays).
* Maintain distribution realism — augmentations should resemble likely real-world variation.

### 1.5 Beyond Images (Quick Reference)

* Text/NLP: Synonym replacement, back-translation, token dropout, EDA, paraphrasing.
* Audio: Time stretch, pitch shift, background noise, SpecAugment (time/freq masking on spectrograms).
* Tabular: SMOTE/ADASYN for minority classes, noise injection, mixup in feature space.

### 1.6 Minimal Code Patterns

Keras (TensorFlow):

```python
from tensorflow.keras import layers, Sequential

augment = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
# model.fit(train_ds.map(lambda x,y: (augment(x, training=True), y)), ...)
```

PyTorch:

```python
import torchvision.transforms as T

train_tfms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.ToTensor(),
])
```

---

## 2) Image Generation with Neural Networks

### 2.1 GANs (Generative Adversarial Networks)

* Players: Generator (G) maps noise z→image; Discriminator (D) classifies real vs fake.
* Training: Min–max game; G tries to fool D, D tries not to be fooled → both improve.
* Conditioning: cGAN / AC-GAN add class labels/attributes; enables class-controlled synthesis.
* Challenges: Mode collapse, unstable training, sensitivity to hyperparameters.
* Stabilizers: Feature matching, spectral normalization, WGAN/WGAN-GP (Earth-Mover distance + gradient penalty), two-time-scale updates, balanced architectures.

### 2.2 VAEs (Variational Autoencoders)

* Structure: Encoder x→(mu, sigma) produces a latent distribution; Decoder samples z \~ Normal(mu, sigma) then reconstructs x\_hat.
* Objective (ELBO): Reconstruction loss + KL-divergence pulling latent toward prior Normal(0, I).
* Pros: Smooth latent space, easy interpolation/sampling; stable training.
* Variants: beta-VAE (disentanglement), VQ-VAE (discrete latents), conditional VAEs.
* Trade-off: Samples sometimes blurrier than GANs due to likelihood objective.

### 2.3 Diffusion Models (DDPM/Score-based)

* Core idea: Learn to denoise step-by-step, reversing a gradual noising process from data → Gaussian noise.
* Sampling: Start from noise; iteratively apply learned denoiser to get an image; DDIM for faster sampling.
* Guidance: Classifier guidance or classifier-free guidance to steer outputs.
* Pros: Very high sample quality, stable training.
* Cons: Many sampling steps (slower), though modern samplers reduce steps drastically.

When to use what:

* GANs: When ultra-sharp images and speed at inference matter, and you can stabilize training.
* VAEs: When you want a structured latent space, reconstructions, anomaly scores.
* Diffusion: When you want state-of-the-art fidelity/coverage and can afford compute.

---

## 3) Neural Network Diagram (MLP)

```
Input Layer            Hidden Layer(s)                     Output Layer
 [x1 x2 ... xn] ---> [h1, h2, ..., hm]  ---> ... --->  [y1, y2, ..., yk]
       |            (W·x + b) + activation                (W·h + b) → softmax
       v
   Linear algebra → Nonlinear transform (e.g., ReLU) → Probabilities/scores
```

Neuron: out = phi(sum\_i (w\_i \* x\_i) + b), where phi is ReLU/sigmoid/tanh/etc.

---

## 4) Overfitting and Dropout Regularization

### 4.1 Overfitting

* Definition: Memorizing noise/peculiarities of the training set → poor validation/test performance (low train loss, high val loss).

### 4.2 Mitigation Toolkit

* Dropout: Randomly zeroes activations during training; forces redundancy.
* BatchNorm: Stabilizes activation distributions; acts as mild regularizer.
* Early stopping: Halt when validation loss stops improving.
* Weight penalties: L1 (sparsity), L2/weight decay (shrinkage).
* Simplify model: Fewer layers/parameters; architectural constraints.
* Data strategies: Augmentation, cross-validation, more data.

### 4.3 Dropout Mechanics

* Training: With keep-prob q = 1 - p, each activation a becomes a\_tilde = m ⊙ a with m \~ Bernoulli(q). Many frameworks use inverted dropout: scale by 1/q at training so no scaling is needed at test.
* Inference: Use full network; if not using inverted dropout, scale activations by q (or equivalently divide by 1 - p during training).

---

## 5) Autoencoders and Data Compression (vs PCA)

### 5.1 Autoencoders (AE)

* Encoder: x → z (compressed representation).
* Decoder: z → x\_hat (reconstruction).
* Uses: Denoising (Denoising AE), anomaly detection (large reconstruction error), feature learning.
* Variants: Sparse AE (L1 on activations), Contractive AE (Jacobian penalty), VAE (probabilistic), Sequence AE.

### 5.2 PCA vs Autoencoder

| Feature      | PCA             | Autoencoder                                        |
| ------------ | --------------- | -------------------------------------------------- |
| Nature       | Linear subspace | Nonlinear manifold (deep nets)                     |
| Custom Loss  | No              | Yes (flexible objectives)                          |
| Supervision  | Unsupervised    | Unsupervised (plus semi/fully supervised variants) |
| Compression  | Yes             | Yes                                                |
| Complex Data | Limited         | Strong (with depth/conv layers)                    |

### 5.3 Loss Example (MSE)

Given original x = \[1, 3, 2], reconstructed x\_hat = \[1.5, 2.5, 2]:
MSE = (1/3) \* ((1-1.5)^2 + (3-2.5)^2 + (2-2)^2) = (1/3) \* (0.25 + 0.25 + 0) = 0.1667.

### 5.4 Anomaly Detection

* Train AE on normal data → it learns the normal manifold.
* Anomalies reconstruct poorly → large error → flag as anomaly.

---

## 6) Supervised vs Unsupervised Learning

* Supervised (labels available): Image classification, fraud detection; loss: cross-entropy, focal.
* Unsupervised (no labels): AEs, clustering embeddings, self-supervised pretraining.
* Semi/Weakly-Supervised: Limited labels + lots of unlabeled data; pseudo-labeling, consistency losses.

---

## 7) Activation Functions (Sigmoid vs Softmax; Role of ReLU)

### 7.1 Sigmoid vs Softmax

* Sigmoid: sigma(x) = 1 / (1 + exp(-x)) → outputs in (0,1). Binary output or independent multi-label heads.
* Softmax: y\_i = exp(x\_i) / sum\_j exp(x\_j) → probability distribution over mutually exclusive classes.

### 7.2 Why ReLU (and friends)?

* ReLU: max(0, x) → non-saturating for x > 0, cheap, sparse activations.
* Leaky ReLU/Parametric ReLU: Fix dead neurons.
* GELU/Swish: Smooth, often better in Transformers/CNNs.

### 7.3 Comparison Table

| Function | Formula                    | Range        | Derivative (key)                  | Notes               |
| -------- | -------------------------- | ------------ | --------------------------------- | ------------------- |
| Sigmoid  | 1/(1+e^(-x))               | (0,1)        | sigma(x) \* (1 - sigma(x)) ≤ 0.25 | Vanishes at tails   |
| Tanh     | tanh(x)                    | (-1,1)       | 1 - tanh(x)^2                     | Zero-centered       |
| ReLU     | max(0,x)                   | \[0,∞)       | 0 (x<0), 1 (x>0)                  | Simple & robust     |
| Softmax  | exp(x\_i)/sum\_j exp(x\_j) | (0,1), sum=1 | Full Jacobian                     | Multi-class outputs |

---

## 8) Loss Functions (When/Why)

* MSE: (1/n) \* sum (y - y\_hat)^2 → regression; penalizes large errors strongly.
* MAE: (1/n) \* sum |y - y\_hat| → regression; robust to outliers.
* Binary Cross-Entropy: -\[t log y + (1 - t) log (1 - y)] with sigmoid.
* Categorical Cross-Entropy: -sum\_i t\_i log y\_hat\_i with softmax.
* Hinge/Focal: Margin-based or class-imbalance-aware (focal) for detection/rare classes.
* Label smoothing: Replace one-hot with softened targets to improve calibration.

Pairing rule-of-thumb:

* Sigmoid ↔ BCE (binary/multi-label).
* Softmax ↔ Categorical CE (mutually exclusive classes).

---

## 9) Why Nonlinear Activations Are Necessary

If phi(x) = x everywhere (purely linear), any stack of layers reduces to a single linear map: y = W3(W2(W1 x)) = (W3 W2 W1) x. This cannot model XOR, curvature, or complex boundaries. Nonlinearity enables universal approximation on compact sets.

---

## 10) One-vs-Rest (OvR) Multiclass with Vector Output

* Task: Fruits {apple, mango, papaya, litchi, banana, dragon-fruit}.
* Head: 6 output neurons, each sigmoid → independent “is class k?” scores.
* Loss: Sum of binary cross-entropies vs one-hot target. Works for multi-label too (multiple 1s).
* Alternative: Single softmax head (mutually exclusive); typically preferred for multiclass.

---

## 11) Initialization, Backprop, Skip Connections, Gradient Flow

### 11.1 Symmetry Breaking

* All weights equal → identical neuron outputs and gradients → no diversity.
* Use random init (e.g., Xavier/Glorot for tanh/sigmoid, He/Kaiming for ReLU family) to match variance across layers.

### 11.2 Backprop (Chain Rule)

* Local gradients multiply along paths from loss to parameters. Each layer needs only its local derivative and the upstream gradient.
* With identity activations, local derivative = 1, so gradients become products of weights along the path.

### 11.3 Residual/Skip Connections

* Form: h\_{l+1} = F(h\_l) + h\_l.
* Benefit: Direct gradient highways back to earlier layers → mitigates vanishing, enables very deep nets (ResNets).

### 11.4 Normalization and Regularization

* BatchNorm/LayerNorm/GroupNorm improve conditioning and gradient flow; combine with dropout and weight decay (AdamW) thoughtfully.

---

## 12) Dropout Numericals (Worked Examples)

Let drop prob = p, keep prob = q = 1 - p. Using test-time scaling perspective (or inverted-dropout equivalence):

a) 10 neurons, p = 0.2

* Expected active: 10 \* q = 10 \* 0.8 = 8.
* If a neuron’s activation is 5 (train-time), test-time scaled activation = 5 / q = 5 / 0.8 = 6.25.

b) 50 neurons, p = 0.4

* Keep prob q = 0.6; scaling factor 1/q = 1/0.6 ≈ 1.67.
* Expected active: 50 \* 0.6 = 30.

c) 8 inputs, p = 0.25

* Expected dropped: 8 \* 0.25 = 2; expected active: 6.
* If each active input produced 4 pre-dropout, test-time equivalent per-input = 4 / q = 4 / 0.75 ≈ 5.33.
* Total contribution (expected): 6 \* 5.33 ≈ 32.

d) 4 neurons, p = 0.5; only #1 and #4 active; each outputs 2

* Test-time scaled value per active neuron: 2 / q = 2 / 0.5 = 4.

e) 20 neurons, p = 0.3; exactly 14 active

* Probability of exactly 14 survivors:  C(20,14) \* (0.7)^14 \* (0.3)^6 ≈ 0.1916 (19.16%).

---

## 13) Optimization Landscapes and Algorithms

### 13.1 Geometry Intuition

* Plateaus: Very small gradients → slow progress; try warmup, adaptive methods, LR schedules.
* Cliffs: Very steep regions → overshooting; use gradient clipping, smaller LR.
* Ravines: Curved valleys → oscillation; momentum and adaptive optimizers help.

### 13.2 Algorithms (Core Equations)

* Vanilla GD: theta\_{t+1} = theta\_t - eta \* grad L(theta\_t).
* Momentum: v\_{t+1} = gamma \* v\_t + eta \* grad L(theta\_t); theta\_{t+1} = theta\_t - v\_{t+1}.
* RMSProp: s\_{t+1} = rho \* s\_t + (1-rho) \* (grad L)^2; update with eta / sqrt(s\_{t+1} + eps).
* Adam (with bias-correction):
  m\_{t+1} = beta1 \* m\_t + (1 - beta1) \* g\_t;  v\_{t+1} = beta2 \* v\_t + (1 - beta2) \* g\_t^2.
  m\_hat = m\_{t+1} / (1 - beta1^(t+1)); v\_hat = v\_{t+1} / (1 - beta2^(t+1)).
  theta\_{t+1} = theta\_t - eta \* m\_hat / sqrt(v\_hat + eps).
* AdamW: Adam with decoupled weight decay (better regularization).

### 13.3 Practical Tips

* Start with Adam/AdamW, then fine-tune with SGD+Momentum for extra generalization (common in vision).
* Use cosine decay or OneCycle LR schedules; add warmup for Transformers.
* Clip gradients in RNNs/Transformers to prevent exploding gradients.

---

Quick Recap

* Augmentation expands data realism; choose transforms that preserve semantics.
* Generative models: GAN (adversarial), VAE (latent probabilistic), Diffusion (denoising steps).
* Regularization: Dropout/weight decay/BatchNorm + good init + residual paths.
* Loss/Activations pairing matters (sigmoid↔BCE, softmax↔CE).
* Optimization: Use momentum/adaptive methods; manage LR and gradient norms.
