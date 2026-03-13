# **Handwritten Digit Recognition with CNN**
> A Deep Learning project using TensorFlow and Keras to classify handwritten digits (0–9) with high accuracy using the MNIST dataset.

---

## **📖 Project Details**
This project was developed to explore the power of **Convolutional Neural Networks (CNNs)** in Computer Vision. While traditional machine learning models treat images as flat vectors, this project uses spatial awareness to detect patterns like edges, curves, and loops, which are essential for recognizing handwriting.

### **The Problem**
Handwritten digits vary significantly in style, thickness, and alignment. A robust system must be able to generalize across these variations without being "distracted" by noise or slight shifts in the image.

### **The Solution**
By utilizing **TensorFlow** and the **MNIST dataset** (a collection of $70,000$ grayscale images), this project builds a multi-layered neural network that:
1.  **Normalizes** pixel data to a $0–1$ range for faster convergence.
2.  **Extracts Features** using specialized filters (kernels).
3.  **Reduces Dimensionality** via Max Pooling to keep the model efficient.
4.  **Prevents Overfitting** using Dropout layers, ensuring the model doesn't just "memorize" the training data.

---

## **✨ Key Features**
* **Automated Data Pipeline:** Loads, normalizes, and reshapes the MNIST dataset for optimal training performance.
* **Optimized CNN Architecture:** Uses a combination of `Conv2D`, `MaxPooling2D`, and `Dropout` layers.
* **Real-time Visualization:** Generates accuracy/loss plots to monitor training progress.
* **Randomized Prediction Engine:** Includes a utility function to pick random test images and visualize the model's predictions against actual labels.

---

## **🛠️ Tech Stack**
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow 2.x, Keras
* **Data Science:** NumPy
* **Visualization:** Matplotlib

---

## **🏗️ Model Architecture**
The model follows a sequential flow designed for efficiency:

| Layer (type) | Output Shape | Param # | Description |
| :--- | :--- | :--- | :--- |
| **Conv2D** | (26, 26, 32) | 320 | Detects basic edges/features |
| **MaxPooling2D** | (13, 13, 32) | 0 | Reduces spatial dimensions |
| **Conv2D** | (11, 11, 64) | 18,496 | Detects complex shapes |
| **Dropout (0.25)** | (5, 5, 64) | 0 | Prevents overfitting |
| **Flatten** | (1600) | 0 | Flattens 2D data to 1D |
| **Dense (128)** | (128) | 204,928 | Fully connected layer |
| **Dropout (0.5)** | (128) | 0 | Extra regularization |
| **Dense (10)** | (10) | 1,290 | Output (Softmax probabilities) |

---

## **🚀 Getting Started**

### **Prerequisites**
You will need Python installed along with the following libraries:
```bash
pip install tensorflow matplotlib numpy opencv-python
```
## Execution
Simply run the script to start the training process and view the results:
```bash
python digit_recognition.py
```

## 📊 Results & Visualization
The script automatically produces a visualization of the training history and a sample of predictions:
- **Final Test Accuracy:** Typically achieves approximately 99.1%
- **Validation Plot:** Displays the convergence of Training vs. Validation accuracy.
- **Prediction Grid:** Shows 5 random digits from the test set with their predicted vs. actual labels.

## 🔮 Future Extensions: Testing Your Own Data
Want to see the model predict your handwriting? You can extend this project by adding a function to process external images.

### How to prepare your own image:
1. Draw a digit in a simple paint tool (white digit on a black background).
2. Save it as a 28×28 pixel image.
3. Load and Predict using the following logic:
```python
import cv2
import numpy as np

def predict_custom_image(image_path, model):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to match MNIST dimensions
    img = cv2.resize(img, (28, 28))
    # Normalize and reshape
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img)
    print(f"Predicted Digit: {np.argmax(prediction)}")
```
