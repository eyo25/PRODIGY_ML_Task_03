import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import joblib
# Constants
IMG_SIZE = 32  # Resize images to 32x32 pixels
TRAIN_DIR = r'C:\Users\eyosi\OneDrive\Desktop\ProdigyInfotech\Task3\PRODIGY_ML_Task_03\dogs-vs-cats\train\train'  # Update this path
CATS_LABEL = 0
DOGS_LABEL = 1

# Function to Load Images and Labels
def load_images_and_labels(folder, img_size=IMG_SIZE):
    images = []
    labels = []
    
    # Iterate over all images in the folder
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        
        # Assign label based on filename
        if 'dog' in filename:
            label = DOGS_LABEL
        elif 'cat' in filename:
            label = CATS_LABEL
        else:
            continue  # Skip files that don't match 'cat' or 'dog'
        
        # Read and process image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            # Resize the image to 32x32
            img_resized = cv2.resize(img, (img_size, img_size))
            images.append(img_resized.flatten())  # Flatten to 1D array
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Extract HOG Features from Images
from skimage.feature import hog

def extract_hog_features(images, img_size=32):
    hog_features = []
    for img in images:
        # Reshape the flattened image to 32x32
        img_reshaped = img.reshape(img_size, img_size)
        print(f"Image shape: {img_reshaped.shape}")  # Debugging output
        fd = hog(img_reshaped, orientations=9, pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

# Main workflow
if __name__ == '__main__':
    # Step 1: Load the dataset
    print("Loading training data...")
    X, y = load_images_and_labels(TRAIN_DIR)
    
    # Normalize pixel values (0-255 to 0-1)
    X = X / 255.0
    print(f"Loaded {X.shape[0]} images for training.")
    
    # Step 2: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Extract HOG features from training and test data
    print("Extracting HOG features...")
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # Step 4: Train the SVM classifier with RBF kernel
    print("Training the SVM (RBF) classifier...")
    svm_rbf_hog = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_rbf_hog.fit(X_train_hog, y_train)
    

    # Step 5: Evaluate the model
    y_pred_hog = svm_rbf_hog.predict(X_test_hog)
    accuracy_hog = accuracy_score(y_test, y_pred_hog)
    print(f"SVM (RBF + HOG) Accuracy: {accuracy_hog * 100:.2f}%")
    print("Classification Report for SVM (RBF + HOG):")
    print(classification_report(y_test, y_pred_hog, target_names=['Cat', 'Dog']))

   # Extract HOG features
X_hog = extract_hog_features(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

# Train the SVM model with RBF kernel
svm_clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm_clf, 'svm_hog_model.pkl')

# Evaluate the model
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")