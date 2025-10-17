# 🐸 Bioacoustic CNN Classifier  
### Automatic Classification of Anuran Family Calls Using Convolutional Neural Networks

---

## 📘 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify frog calls into their respective anuran families based on preprocessed bioacoustic features.  
The model was trained and evaluated using spectrogram or feature-extracted data from publicly available datasets.

### Workflow Includes:
- Data loading and preprocessing  
- CNN model definition using TensorFlow/Keras  
- Model training and validation  
- Evaluation and visualization of results  
- Model saving for future inference  

---

## 🧩 Folder Structure

bioacoustic_project/
│
├── bioacoustic_cnn.py # Main script for model training and evaluation
├── requirements.txt # Dependencies list
├── venv/ # Virtual environment (auto-created)
├── anuran_family_classifier.keras # Saved trained model
├── confusion_matrix.png # Confusion matrix visualization
├── training_history.png # Accuracy and loss curves
├── label_classes.npy # Encoded class label mapping
└── dataset/ # (Optional) Folder for input features or audio data

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Create a Virtual Environment
```bash
python -m venv venv
2️⃣ Activate the Environment
Windows (PowerShell):

```bash
Copy code
venv\Scripts\activate
macOS/Linux:

bash
Copy code
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

bash
Copy code
pip install tensorflow numpy pandas scikit-learn matplotlib
🚀 Running the Model
Run the main training script:

bash
Copy code
python bioacoustic_cnn.py
During execution, the terminal will display:

Epoch-wise training and validation accuracy/loss

Final test accuracy and classification report

Example output:

csharp
Copy code
Test Accuracy: 99.10%
Model saved as anuran_family_classifier.keras
Confusion matrix saved as confusion_matrix.png
Training history saved as training_history.png
📈 Results Summary
Metric	Value
Training Accuracy	~98.9%
Validation Accuracy	~98.6%
Test Accuracy	99.1%

Classification Report (macro avg):

Metric	Score
Precision	0.99
Recall	0.99
F1-score	0.99

The model demonstrates strong generalization and reliability across all classes.

🧠 Key Notes
Implemented using TensorFlow 2.x / Keras API

Uses Sequential CNN architecture with convolution, pooling, and dense layers

Model outputs are saved for future reuse or inference

Compatible with CPU and GPU environments
