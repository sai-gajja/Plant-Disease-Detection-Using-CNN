# 🌿 Plant Disease Detection using CNNs
This repository presents an AI-driven web-based application that utilizes Convolutional Neural Networks (CNNs) to detect diseases in plant leaves. With an accuracy of up to 98.44%, this model aims to assist farmers in early plant disease identification, helping to reduce crop losses and improve agricultural productivity.

📌 Table of Contents
🎯 Project Overview

🧠 Model Architecture

🌱 Dataset Details

⚙️ Installation

🚀 Usage

📊 Results

🔮 Future Scope

🤝 Contributors

📄 License

🎯 Project Overview
This project focuses on:

Early detection of plant leaf diseases using CNNs.

Deploying a user-friendly web interface to upload and diagnose leaf images.

Offering fertilizer and pesticide suggestions post-detection.

Promoting sustainable agriculture through technological innovation.

🧠 Model Architecture
The model is a deep CNN with:

Multiple convolution, pooling, and dropout layers

Followed by fully connected dense layers for classification

Performance:
Training Accuracy: 98.44%

Validation Accuracy: 96.25%

Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix

🌱 Dataset Details
Source: PlantVillage Dataset

Total Images: 86,000+

Classes: 33 disease labels across 14 plant species

Image Size: 128 × 128

Preprocessing Steps:
Normalization

Data Augmentation: Flip, Zoom, Rotate

Class Balancing

⚙️ Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/plant-disease-detection-cnn.git
cd plant-disease-detection-cnn

# Install required packages
pip install -r requirements.txt
🚀 Usage
Run the training notebook or load the pre-trained model.

Launch the web interface:

bash
Copy
Edit
streamlit run app.py
Upload a leaf image through the interface.

View the disease classification and recommended treatment.

📊 Results
Confusion Matrix: Shows strong confidence across most disease classes.

GUI/Web Interface Features:

Upload image

Get prediction with treatment suggestions

See classification result with explanation

🔮 Future Scope
Incorporate adaptive learning using user feedback.

Include environmental data such as weather and soil conditions.

Add pest detection and crop yield prediction.

Deploy to mobile or drone-based platforms for real-time diagnosis.
