🌿 Plant Disease Detection using CNNs


This repository presents an AI-driven web-based application that utilizes Convolutional Neural Networks (CNNs) to detect diseases in plant leaves. With an accuracy of up to 98.44%, this model aims to assist farmers in early plant disease identification, helping to reduce crop losses and improve agricultural productivity.

📌 Table of Contents
Demo

Project Overview

Model Architecture

Dataset Details

Installation

Usage

Results

Future Scope

Contributors

License

🎯 Project Overview
This project focuses on:

Early detection of plant leaf diseases using CNNs.

Deploying a user-friendly web interface to upload and diagnose leaf images.

Offering fertilizer and pesticide suggestions post-detection.

Promoting sustainable agriculture through technological innovation.

🧠 Model Architecture
The model is a deep CNN with multiple convolution, pooling, and dropout layers followed by dense layers for classification.


Training Accuracy: 98.44%

Validation Accuracy: 96.25%

Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix

🌱 Dataset Details
Source: PlantVillage Dataset (includes Apple, Tomato, Grape, Potato, Corn, etc.)

Total Images: 86,000+

Classes: 33 labels across 14 plant species

Image Size: 128×128

Preprocessing:

Normalization

Augmentation (flip, zoom, rotate)

Class balancing


📊 Results
Confusion Matrix: Shows strong classification confidence across most disease classes.

GUI/Web Interface:

Upload image

Get prediction with treatment recommendations

See classification result with explanation


🔮 Future Scope
Incorporate adaptive learning with user feedback.

Add environmental factors (e.g., weather, soil conditions).

Expand to include pest detection and crop yield prediction.

Deploy to mobile or drone-based platforms for real-time diagnosis.
