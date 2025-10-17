# 🐸 Bioacoustic CNN Classifier
### Automatic Classification of Anuran Family Calls Using Convolutional Neural Networks

---

## 📘 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify frog calls into their respective anuran families using pre-processed bioacoustic features.  
The model is trained and evaluated on publicly available feature datasets (e.g., MFCCs).

**Workflow**
- Data loading and preprocessing  
- CNN definition with TensorFlow / Keras  
- Training and validation  
- Evaluation and visualization  
- Saving the trained model for later inference  

---

## ⚙️ Installation and Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
```

### 2️⃣ Activate Environment

**Windows (PowerShell):**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 Running the Model
```bash
python bioacoustic_cnn.py
```

During training you’ll see:
* Epoch-wise accuracy and loss  
* Validation metrics  
* Final test accuracy and classification report  

**Example Output**
```
Test Accuracy: 99.10%
Model saved as anuran_family_classifier.keras
Confusion matrix saved as confusion_matrix.png
Training history saved as training_history.png
```

---

## 📈 Results Summary

| Metric              |    Value   |
| :------------------ | :--------: |
| Training Accuracy   |  ≈ 98.9 %  |
| Validation Accuracy |  ≈ 98.6 %  |
| Test Accuracy       | **99.1 %** |

**Classification Report (macro avg)**

| Metric    | Score |
| :-------- | :---: |
| Precision |  0.99 |
| Recall    |  0.99 |
| F1-score  |  0.99 |

The model shows strong generalization and reliability across classes.

---

## 🧠 Key Notes

* Built with **TensorFlow 2.x / Keras API**
* Sequential CNN with Conv1D, Pooling, Dense, Dropout layers
* Model and plots automatically saved after training
* Runs on CPU or GPU

---
