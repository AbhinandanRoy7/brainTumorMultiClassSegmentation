# 🧠 Brain Tumor MRI Analyzer

### Multi-Class Segmentation & Visualization System

---

## 🚀 Overview

Brain Tumor MRI Analyzer is an **end-to-end medical AI system** that performs **multi-class brain tumor segmentation** on MRI scans and visualizes tumor regions in real time.

The system supports multiple input formats and provides:

* 🎯 Pixel-wise tumor segmentation
* 🧩 Multi-class tumor region identification
* 📊 Confidence estimation & tumor metrics
* 🖥️ Interactive web-based visualization

---

## 🧠 Problem Statement

Manual MRI analysis is:

* ⏱️ Time-consuming
* ❌ Error-prone
* 🧩 Difficult for non-experts

This project automates tumor detection and provides **clear visual insights** in seconds.

---

## ✨ Key Features

### 🔍 Multi-Class Segmentation

Detects and classifies tumor regions into:

| Class | Description          |
| ----- | -------------------- |
| 0     | Background           |
| 1     | Edema (WT)           |
| 2     | Tumor Core (TC)      |
| 3     | Enhancing Tumor (ET) |

---

### 🎨 Visual Overlay

* Color-coded tumor regions:

  * 🔵 Edema
  * 🟡 Tumor Core
  * 🔴 Enhancing Tumor
* Overlay on MRI for intuitive understanding

---

### 📊 Intelligent Insights

* Tumor presence detection
* Tumor location (Left / Right / Central)
* Confidence score
* Tumor area (%)
* Class-wise composition

---

### 📂 Multi-Format Support

Handles:

* ✅ `.h5` (training format)
* ✅ `.nii / .nii.gz` (BraTS MRI volumes)
* ✅ `.jpg / .png` (fallback heuristic mode)

---

### 🧠 Adaptive Inference

* Probability smoothing for better edema detection
* Dynamic thresholding to improve segmentation coverage
* Robust handling of out-of-distribution inputs

---

## 🏗️ System Architecture

```text
Input MRI
   ↓
Preprocessing (slice extraction, normalization)
   ↓
U-Net (ResNet50 Encoder)
   ↓
Multi-class segmentation (0–3)
   ↓
Post-processing & smoothing
   ↓
Visualization + Metrics
   ↓
Web UI (Flask + JS)
```

---

## ⚙️ Tech Stack

### 🔹 Machine Learning

* PyTorch
* segmentation_models_pytorch (SMP)
* NumPy

### 🔹 Backend

* Flask (REST API)

### 🔹 Frontend

* HTML / CSS / Vanilla JavaScript
* Base64 image rendering

---

## 🧠 Model Details

* **Architecture:** U-Net
* **Encoder:** ResNet50
* **Input:** 4-channel MRI (T1, T1ce, T2, FLAIR)
* **Output:** 4-class segmentation mask

---

### 📉 Loss Function

To address class imbalance:

```python
Loss = 0.5 * CrossEntropy (weighted) + 0.5 * Dice Loss
```

Class weights:

```text
[0.05, 0.3, 0.3, 0.35]
```

---

### 🛠️ Optimization Techniques

* Weighted loss for imbalance handling
* Gradient clipping for stability
* Learning rate scheduling
* Data augmentation (flip, rotation)
* Probability threshold tuning for edema

---

## 📊 Example Output

* Segmented tumor regions highlighted on MRI
* Dice scores (evaluation mode)
* Tumor area distribution
* Confidence estimation

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-analyzer.git
cd brain-tumor-analyzer
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Backend

```bash
python app.py
```

---

### 4️⃣ Open the Web Interface

```
http://localhost:5000
```

---

## 📂 Project Structure

```text
├── data/                 # H5 training data
├── models/               # Saved model weights
├── model.py              # Model architecture
├── train.py              # Full training pipeline
├── train_quick.py        # Fast CPU training
├── pipeline.py           # Inference pipeline
├── app.py                # Flask backend
├── static/               # Frontend assets
└── templates/            # HTML UI
```

---

## 📈 Performance

| Metric         | Score        |
| -------------- | ------------ |
| Overall Dice   | ~0.75        |
| WT (Edema)     | Improving    |
| TC (Core)      | High (~0.78) |
| ET (Enhancing) | High (~0.79) |

---

## ⚠️ Limitations

* 2D slice-based segmentation (no full 3D context)
* Heuristic fallback for non-MRI images
* Confidence score is approximate (not clinical)

---

## 🔮 Future Improvements

* 3D U-Net / UNETR integration
* True tumor type classification
* Advanced post-processing
* Cloud deployment

---

## 💡 Key Learnings

* Handling **class imbalance in medical datasets**
* Importance of **loss design (Dice + CE)**
* Real-world ML requires **post-processing & heuristics**
* Bridging ML models with **usable products**

---

## 👨‍💻 Author

**Abhinandan Roy**

* Machine Learning Enthusiast
* Interested in AI for Healthcare

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!

---
