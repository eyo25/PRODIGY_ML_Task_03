
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from skimage.feature import hog

# Load the trained SVM model
model = joblib.load('svm_hog_model.pkl')

# Function to preprocess image and extract HOG features
def preprocess_image(image, img_size=32):
    # Convert image to grayscale and resize
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((img_size, img_size))  # Resize image to 32x32 pixels
    img_array = np.array(img)  # Convert to NumPy array

    # Extract HOG features
    features = hog(img_array, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), visualize=False)
    
    return np.array(features).reshape(1, -1)  # Return the features as a 1D array for prediction

# Define the Streamlit app
def main():
    st.title("Cat vs Dog Classifier using SVM (RBF + HOG)")
    
    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Step 1: Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Step 2: Preprocess the image (resize + HOG feature extraction)
        img_features = preprocess_image(image)
        
        # Step 3: Make predictions using the trained model
        prediction = model.predict(img_features)
        
        # Step 4: Display the result
        if prediction[0] == 0:
            st.write("Prediction: Cat 🐱")
        else:
            st.write("Prediction: Dog 🐶")

# Run the app
if __name__ == '__main__':
    main()
>>>>>>> bb67d25e6cf35b3ec832ea3b9f7f688a9c574827
