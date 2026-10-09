# Real-Time Facial Emotion Detection Using CNN and Transfer Learning

## 📌 Overview

This project detects human faces through a webcam and classifies facial expressions into five categories: **Angry, Happy, Neutral, Sad, and Surprise**.

It implements a Convolutional Neural Network (CNN) for emotion classification and Transfer Learning using MobileNetV2. OpenCV is used for real-time face detection and video processing.

## 🚀 Features

* Real-time facial emotion detection using a webcam.
* Face detection using OpenCV Haar Cascade.
* CNN-based facial emotion classification.
* Transfer Learning using pretrained MobileNetV2.
* Supports five emotion categories.
* Displays predicted emotions on detected faces.

## 🛠️ Technologies Used

* Python
* TensorFlow and Keras
* OpenCV
* NumPy
* CNN (Convolutional Neural Network)
* MobileNetV2 (Transfer Learning)

## 🧠 Models

**1. CNN Model**

* Input size: 48 × 48 grayscale images.
* Uses convolutional, max-pooling, flattening, and dense layers.
* Optimizer: Adam.
* Loss function: Categorical Cross-Entropy.
* Output: Five emotion classes.

**2. Transfer Learning Model**

* Pretrained MobileNetV2 with ImageNet weights.
* Input size: 96 × 96 RGB images.
* Frozen pretrained layers with a custom classification head.
* Dropout for regularization.

## 📂 Project Structure

```text
Emotion-Detection/
├── emotion_detection.py
├── transfer_learning.py
├── cam.py
├── modelGG.json
├── model1GG.weights.h5
└── README.md
```

* `emotion_detection.py` — Trains the CNN model.
* `transfer_learning.py` — Trains the MobileNetV2-based model.
* `cam.py` — Detects faces and predicts emotions in real time.
* `modelGG.json` — Stores the CNN architecture.
* `model1GG.weights.h5` — Stores the trained CNN weights.

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Emotion-Detection
```

Install the dependencies:

```bash
pip install tensorflow opencv-python numpy
```

Organize your dataset into five folders: `angry`, `happy`, `neutral`, `sad`, and `surprise`. Update the dataset and model paths in the scripts to match your system.

## ▶️ Usage

**Train the CNN model:**

```bash
python emotion_detection.py
```

**Train the Transfer Learning model:**

```bash
python transfer_learning.py
```

**Start real-time emotion detection:**

```bash
python cam.py
```

Press **`q`** to close the webcam window.

## 🔮 Future Improvements

* Improve accuracy using a larger, more diverse dataset.
* Evaluate and compare CNN and MobileNetV2 performance.
* Add prediction confidence scores.
* Develop a user-friendly graphical interface.

## ⚠️ Limitations

Prediction performance depends on lighting, image quality, facial orientation, and dataset diversity. Facial expressions are not definitive indicators of a person's actual emotional state.
