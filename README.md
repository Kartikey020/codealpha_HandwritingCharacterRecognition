# ✍️ Handwritten Character Recognition (MNIST / EMNIST)

## 📌 Overview
This project builds a **Convolutional Neural Network (CNN)** to recognize handwritten digits and characters.  
It supports:
- **MNIST** (digits 0–9)  
- **EMNIST Balanced** (digits + uppercase/lowercase letters, 47 classes)  

---

## 🚀 Features
- Automatic dataset download (MNIST or EMNIST).
- Preprocessing (normalization, reshaping).
- CNN architecture with dropout for generalization.
- Model training & evaluation with test accuracy and classification report.
- Saves trained model (`.h5` file).

---

## 📂 Project Structure
handwritten_character_recognition/
│── data/ # Auto-downloaded datasets
│── notebooks/ # Jupyter notebook for EDA
│── src/ # Source code
│ ├── data_loader.py # Load MNIST/EMNIST
│ ├── preprocessing.py # Normalize & reshape
│ ├── model.py # CNN architecture
│ ├── train.py # Training loop
│ ├── evaluate.py # Evaluation metrics
│ ├── utils.py # Helpers (plot sample, save model)
│── main.py # Run the full pipeline
│── requirements.txt # Dependencies
│── README.md # Project documentation


---

## ⚙️ Installation

git clone https://github.com/yourusername/handwritten_character_recognition.git
cd handwritten_character_recognition
python -m venv venv
venv\Scripts\activate   # (Windows)
source venv/bin/activate # (Linux/Mac)
pip install -r requirements.txt

▶️ Usage

Run with MNIST digits:python main.py
Or switch to EMNIST Balanced by updating main.py to use:from src.data_loader import load_emnist
(X_train, y_train), (X_test, y_test) = load_emnist("balanced")

📊 Example Output:
===== HANDWRITTEN CHARACTER RECOGNITION (EMNIST) =====
Dataset Loaded: (112800, 28, 28, 1) (18800, 28, 28, 1)
Epoch 1/5 ...
Test Accuracy: 0.97xx
Model saved as emnist_cnn.h5

📚 Datasets

MNIST
 (digits 0–9).

EMNIST
 Balanced (47 classes: digits + letters).


 🛠️ Technologies

Python
NumPy, Pandas
TensorFlow / Keras
TensorFlow Datasets (TFDS)
Matplotlib
