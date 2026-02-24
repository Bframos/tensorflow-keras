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
* **GPU Integration:** Verifying and utilizing NVIDIA hardware for faster processing.



## 🛠️ Setup and Installation

To run the code in this repository, you need Python and the TensorFlow library installed:

```bash
# Install the environment
pip install tensorflow jupyter

# Check if GPU is recognized (optional)
nvidia-smi
```
## 🖼️ Case Study: Training on CIFAR-10 with Keras Layers

After understanding Tensors and Gradients, we apply these concepts using **Keras Layers** to solve a real-world computer vision problem: classifying images from the **CIFAR-10** dataset (60,000 32x32 color images in 10 classes).

### 🏗️ Building the Model with Keras
We use a **Sequential** stack of layers to build a Convolutional Neural Network (CNN):

* **Conv2D & MaxPooling2D:** Used for feature extraction (detecting edges, shapes, and textures).
* **Flatten:** Converts the 2D feature maps into a 1D vector.
* **Dense (Fully Connected):** Interprets the features to perform the final classification.
* **Dropout:** A regularization technique to prevent overfitting.



### 🧪 Training Process
Using the high-level Keras API, the training workflow is simplified:
1.  **Normalization:** Scaling pixel values from [0, 255] to [0, 1].
2.  **Compilation:** Defining the `adam` optimizer and `sparse_categorical_crossentropy` loss function.
3.  **Fitting:** Training the model for several epochs.

### 📈 Results

The notebook includes visualizations showing the evolution of **loss** and **accuracy** on the training and validation sets over the epochs, demonstrating the convergence of the manually implemented algorithm.

| Accuracy | Loss |
|--------|--------|
| ![Accuracy](plots/acc.png) | ![Loss](plots/loss.png) |
