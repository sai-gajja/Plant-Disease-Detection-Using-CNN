# 🌿 Plant Disease Detection using CNNs

This repository presents an **AI-powered web application** that leverages **Convolutional Neural Networks (CNNs)** for **early detection of plant leaf diseases**. Achieving an impressive accuracy of **up to 98.44%**, the model supports **farmers and agronomists** in diagnosing plant health, reducing crop losses, and promoting sustainable farming.

---

## 📌 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [🧠 Model Architecture](#-model-architecture)
- [🌱 Dataset Details](#-dataset-details)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📊 Results](#-results)
- [🔮 Future Scope](#-future-scope)
- [👥 Contributors](#-contributors)
- [📝 License](#-license)

---

## 🎯 Project Overview

This project focuses on:

- 🌿 Early detection of plant diseases through deep learning
- 🌐 Deploying a user-friendly web interface for diagnosis
- 🧪 Suggesting fertilizers and pesticides post-detection
- 🌱 Encouraging sustainable agriculture with technology

---

## 🧠 Model Architecture

The model is a **custom CNN** composed of:

- 📦 Convolutional Layers  
- 🌀 Pooling Layers  
- 🔁 Dropout Layers  
- 🎯 Fully Connected Dense Layers for final classification  

**Performance:**

- ✅ **Training Accuracy:** 98.44%  
- ✅ **Validation Accuracy:** 96.25%  

**Evaluation Metrics:**

- 📈 Precision
- 🔁 Recall
- 📊 F1-Score
- 🔀 Confusion Matrix

---

## 🌱 Dataset Details

- **Source:** PlantVillage Dataset  
- **Total Images:** 86,000+  
- **Classes:** 33 disease types across 14 plant species  
- **Image Size:** 128×128 pixels  

**Preprocessing Includes:**

- 🧼 Normalization  
- 🔁 Augmentation: Flip, Zoom, Rotate  
- ⚖️ Class Balancing  

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
