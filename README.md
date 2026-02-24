# Introduction to TensorFlow and Keras

This repository serves as a practical introduction to **TensorFlow** and **Keras**, covering everything from low-level tensor operations to high-level deep learning concepts.

## 📘 Project Overview

This document is designed to guide you through the fundamental ecosystem of modern Machine Learning. It explains how data is handled at a hardware level and how we can use high-level abstractions to build neural networks.

### 1. TensorFlow: The Engine
**TensorFlow** is the backbone of the project. It is an open-source platform that allows for:
* **Mathematical Operations:** Handling complex multi-dimensional arrays (Tensors).
* **Hardware Acceleration:** Running computations seamlessly on CPUs, GPUs, or TPUs.
* **Automatic Differentiation:** Using `GradientTape` to calculate gradients, which is the core of how AI models "learn".

### 2. Keras: The Interface
**Keras** is the high-level API built on top of TensorFlow. While TensorFlow handles the "heavy lifting," Keras provides:
* **Simplicity:** A user-friendly way to define layers, models, and optimizers.
* **Modularity:** Easily stack layers like `Dense` or `Dropout` to create complex architectures.
* **Efficiency:** It bridges the gap between complex mathematical theory and rapid prototyping.

---

## 🚀 Key Topics Covered

* **Tensors & Variables:** The difference between constant data and trainable parameters (`tf.Variable`).
* **Tensor Operations:** How to create, reshape, and manipulate data structures.
* **Gradient Computation:** Using `tf.GradientTape` to track operations and compute derivatives automatically.
* **GPU Integration:** Verifying and utilizing NVIDIA hardware for faster processing (`nvidia-smi`).
* **Manual Training Loop:** Implementing gradient descent by hand using `GradientTape` — demonstrating what Keras abstracts under the hood.

---

## 🛠️ Setup and Installation

To run the code in this repository, you need Python and the TensorFlow library installed:

```bash
# Install the environment
pip install tensorflow jupyter

# Check if GPU is recognized (optional)
nvidia-smi
```

---

## 🖼️ Case Study: Training on CIFAR-10 with Keras

After understanding Tensors and Gradients, we apply these concepts using **Keras** to solve a real-world computer vision problem: classifying images from the **CIFAR-10** dataset (60,000 32×32 color images in 10 classes).

---

## 🏗️ Building the Model for CIFAR-10

Instead of complex convolutional architectures, we use a **Dense (Fully Connected) Neural Network** to demonstrate the direct flow of data and how weights are optimized.

### 🔄 Data Preprocessing

Before being fed into the model, the images are preprocessed **outside** the model using NumPy:

* The 32×32×3 images are **manually reshaped** into 1D vectors of 3,072 values via `.reshape()`.
* Pixel values are **normalized** from [0, 255] to [0, 1] by dividing by 255.

> ⚠️ No `Flatten` layer is used inside the model — flattening is done as a preprocessing step before training.

---

### 🧠 Model Architecture

The model is built using the **Keras Sequential API** with the following structure:

| Layer | Units | Activation |
|-------|-------|------------|
| `Dense` | 64 | ReLU |
| `Dense` (Output) | 10 | Softmax |

The output layer has **10 units**, one for each CIFAR-10 class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), and uses **Softmax** to output a probability distribution.

---

### 🧪 Training Process

The notebook trains **two equivalent models** to demonstrate different ways of handling labels:

**Model 1 — Integer Labels:**
```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # labels as integers
    metrics=["accuracy"]
)
```

**Model 2 — One-Hot Encoded Labels:**
```python
train_labels_cat = to_categorical(train_labels, num_classes=10)  # convert to one-hot

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # labels as one-hot vectors
    metrics=["accuracy"]
)
```

Both approaches are functionally equivalent — the difference is only in label format.

---

### 📈 Results

The notebook includes visualizations showing the evolution of **loss** and **accuracy** on the training and validation sets over 10 epochs, using Keras' `model.fit()` history object.

| Accuracy | Loss |
|----------|------|
| ![Accuracy](plots/acc.png) | ![Loss](plots/loss.png) |
