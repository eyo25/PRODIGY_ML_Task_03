
# Prodigy ML Task 03: Cat vs Dog Image Classification

This project is a machine learning model that classifies images of cats and dogs using **Histogram of Oriented Gradients (HOG)** features and an **SVM (Support Vector Machine) with an RBF kernel**. The application is deployed using **Streamlit**, allowing users to upload images and get predictions on whether the image is a cat or a dog.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Model Development](#model-development)
   - [Data Preprocessing](#data-preprocessing)
   - [HOG Feature Extraction](#hog-feature-extraction)
   - [Training the SVM Model](#training-the-svm-model)
5. [Streamlit Application](#streamlit-application)
6. [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
7. [Results](#results)

---

## Project Overview

The goal of this project is to classify images of cats and dogs using a machine learning model. We used the **SVM (RBF Kernel)** algorithm trained on **HOG features** to distinguish between the two classes. The model was deployed as a **Streamlit app**, which allows users to upload images and get a prediction (cat or dog) with an accuracy of around **72.24%**.

---

## Technologies Used

- **Python 3.x**
- **OpenCV** (for image processing)
- **scikit-learn** (for model training and evaluation)
- **scikit-image** (for HOG feature extraction)
- **Streamlit** (for web app deployment)
- **Git LFS** (for handling large model files)
- **Google Drive/AWS S3** (for model storage if the file exceeds GitHub limits)

---

## Setup Instructions

### Prerequisites
- Ensure that you have **Python 3.x** installed.
- Install the required dependencies using the provided `requirements.txt`.

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/PRODIGY_ML_Task_03.git
   cd 
   ```
2. Install the required Python packages:ba
   ```bash
   pip install -r requirements.txt
   ```
3.If using Git LFS for large files, ensure that you have Git LFS installed:
```bash
git lfs install
git lfs pull
```
4. Run the Streamlit app locally:
```bash
streamlit run app.py
```

## Model Development
### 1. Data Preprocessing
The dataset contains images of cats and dogs. All images are resized to 32x32 pixels to make them consistent in size and suitable for feature extraction.
The images are converted to grayscale before extracting features.
### 2. HOG Feature Extraction
We use Histogram of Oriented Gradients (HOG) to extract features from each image. HOG captures the gradient and edge information of an image, which is helpful for distinguishing between the shapes of cats and dogs.
### 3. Training the SVM Model
We train a Support Vector Machine (SVM) with an RBF kernel on the HOG features. The RBF kernel allows the model to capture non-linear relationships between the features.
We used GridSearchCV to fine-tune the hyperparameters (C, gamma) for optimal performance. 


## Streamlit Application
The Streamlit app allows users to upload an image, and it will classify whether the image contains a cat or a dog. The app processes the image, extracts HOG features, and uses the trained SVM model to predict the result.

### App Features:
Image upload using Streamlit’s file_uploader.
Image preprocessing (resizing, grayscale conversion).
HOG feature extraction.
Prediction display (Cat 🐱 or Dog 🐶).

## Deployment on Streamlit Cloud
### 1. Push Code to GitHub
Ensure that all your code, along with the requirements.txt and model file (svm_hog_model.pkl), is pushed to GitHub.

### 2. Streamlit Cloud Setup
Log in to Streamlit Cloud.
Select your repository and configure the app entry point (app.py).
Deploy the app by clicking Deploy.
### 3. Handling Large Files
If the model file is too large for GitHub (over 100 MB), use Git LFS to manage large files:
```bash
git lfs install
git lfs track "*.pkl"
git add svm_hog_model.pkl
git commit -m "Add model file using Git LFS"
git push origin main
```
Alternatively, the model can be hosted on Google Drive or AWS S3 and dynamically loaded in the app using requests or boto3.


## Results 
### Model Accuracy
The SVM (RBF + HOG) model achieved an accuracy of 72.24% on the test set.

Classification Report:
```bash

              precision    recall  f1-score   support
         Cat       0.73      0.70      0.72      2515
         Dog       0.71      0.74      0.73      2485
    accuracy                           0.72      5000
   macro avg       0.72      0.72      0.72      5000
weighted avg       0.72      0.72      0.72      5000
```
This model can be improved by increasing the image size or using more advanced techniques like CNNs or Transfer Learning.

 ## Conclusion
This project demonstrates a practical approach to solving image classification problems using Support Vector Machines and HOG features. The Streamlit app makes it easy to deploy the model and interact with users through a web interface. Future improvements could involve using Convolutional Neural Networks (CNNs) for better accuracy and robustness.

