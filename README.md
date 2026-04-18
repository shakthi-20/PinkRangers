# 🩺 PinkRangers — Breast Cancer Screening from Thermal Images

## 📄 Publication

**Breast Cancer Screening from Thermal Images: CNN-Based Analysis with Augmented Datasets and Web App Deployment**  
*IEEE International Conference on Intelligent Signal Processing and Effective Communication Technologies (INSPECT), Gwalior, India — November 2025*  
📎 [View on IEEE Xplore](https://doi.org/10.1109/INSPECT67393.2025.11350372)

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **84%** |
| ROC-AUC Score | **0.92** |
| Recall (Malignant) | **97%** |

---

## 🔍 Overview

Breast cancer diagnosis traditionally relies on invasive procedures. This project proposes a non-invasive CNN-based approach using thermal imaging (thermography) as an alternative screening method.

Key contributions:
- Compared two preprocessing pipelines: Wiener filtering + histogram equalization vs. watershed segmentation
- Found that watershed segmentation **degrades** model performance due to artifact introduction
- Evaluated wavelet-based feature extraction vs. end-to-end deep learning
- Deployed a real-time Streamlit web app for clinical screening support

---

## 🛠️ Tech Stack

- **Deep Learning:** CNN (TensorFlow/Keras)
- **Preprocessing:** Wiener filtering, histogram equalization, Gaussian blur, image rotation
- **Classical ML (comparison):** Wavelet features + One-class SVM
- **Web App:** Streamlit (`PinkRangers.py`)
- **Dataset:** Breast thermography images (augmented)

---

## 📁 Repository Structure

```
PinkRangers/
├── PinkRanger_Model.ipynb        # Model training notebook
├── AUGMENTATION_DATASET.ipynb    # Data augmentation pipeline
├── PinkRangers.py                # Streamlit web application
├── cnn_model_preprocessed.h5     # Trained model weights
├── Breast Thermography.zip       # Original dataset
├── Augmented_Breast_Thermography.zip  # Augmented dataset
└── requirements.txt              # Dependencies
```

---

## 🚀 Run the Web App

```bash
pip install -r requirements.txt
streamlit run PinkRangers.py
```

---

## ⚠️ Disclaimer

This tool is a **diagnostic supplement only** and is not a substitute for professional clinical evaluation. Model generalizability may be limited by dataset size.

---

## 👥 Authors

- **S Shakthi** — [GitHub](https://github.com/shakthi-20)
- Shrinidhi Ganesh
- P Supriya
