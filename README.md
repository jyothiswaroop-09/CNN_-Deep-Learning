<h1>Image Classification</h1>
# 🐶🐱 Cat vs Dog Image Classification

This project is a simple deep learning web app built using **TensorFlow** and deployed with **Streamlit**. It classifies images as either a **Cat** or a **Dog** using a Convolutional Neural Network (CNN) trained on image data.

## 🔍 Project Overview

- **Objective**: Build an image classification model that can predict whether an uploaded image is a cat or a dog.
- **Model**: Convolutional Neural Network (CNN) implemented using **TensorFlow/Keras**.
- **Deployment**: Interactive web app using **Streamlit**.

---

## 📁 Project Structure
```
📦cat-dog-classifier
├── app.py # Streamlit application script
├── model.h5 # Trained CNN model
├── utils.py # Image preprocessing and prediction functions
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── 📁images # Sample cat and dog images for testing
```
## ⚙️ Installation

1. Clone the repository:
git clone https://github.com/jyothiswaroop-09/CNN_-Deep-Learning.git
cd cat-dog-classifier

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate

3.Install the required packages:
pip install -r requirements.txt

## 🚀 Run the App
To start the Streamlit app, run the following command:
streamlit run app.py

## 🧠 Model Details
Architecture: 3-layer CNN with ReLU activations and MaxPooling
Optimizer: Adam
Loss Function: Binary Crossentropy
Accuracy Achieved: ~90% on validation set

## 📷 How to Use
Open the app in your browser.
Upload a clear image of a cat or dog.
The app will display the prediction result with confidence.

## 💡 Future Improvements
Add more animal categories
Use transfer learning for higher accuracy
Deploy on cloud (Streamlit Sharing / Hugging Face Spaces / Heroku)

