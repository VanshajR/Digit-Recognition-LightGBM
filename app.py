import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Load pre-trained model
model = joblib.load("digit_recognition_model.pkl")

# Load PCA for dimensionality reduction
pca = joblib.load("pca_model.pkl") 

# Function to preprocess uploaded image
def preprocess_image(uploaded_image):
    # Open the uploaded image using PIL
    img = Image.open(uploaded_image)
    
    # Convert the image to grayscale
    img = img.convert("L")
    
    # Resize image to 28x28
    img = img.resize((28, 28))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Flatten the image to match the model input (784 pixels) as per the dataset
    img_flattened = img_array.flatten() / 255.0  # Normalize pixel values
    
    # Apply PCA transformation to reduce dimensionality
    img_pca = pca.transform([img_flattened])
    
    return img_pca

# Streamlit app
st.set_page_config(page_icon="🔢", page_title="Digit Recognition App")
st.title("Handwritten Digit Recognition with LightGBM 🔢")
st.write("This is a simple digit recognition app using LightGBM and PCA. Some images from the MNIST dataset can be found on the github repository of this project for testing this application.")
# Upload image
uploaded_image = st.file_uploader("Upload an image of a digit", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and get prediction
    img_pca = preprocess_image(uploaded_image)
    
    # Predict the digit
    prediction = np.argmax(model.predict(img_pca), axis=1)
    st.write(f"Predicted Digit: {prediction[0]}")

    # Evaluate model on the validation set for visualization
    # Load the dataset for evaluation (this can be adapted if you're using the same dataset in the app)
    train_data = pd.read_csv("train.csv")
    X = train_data.drop(columns=["label"]).values / 255.0  # Normalize pixel values
    y = train_data["label"].values

    # Dimensionality reduction (same PCA model)
    X_reduced = pca.transform(X)

    # Predictions
    y_pred = np.argmax(model.predict(X_reduced), axis=1)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Display metrics and confusion matrix
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Accuracy on Validation Set: {accuracy}")
    st.write("Classification Report:")
    st.text(classification_report(y, y_pred))

    # Plot Confusion Matrix
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot()

    # Feature importance visualization
    st.subheader("Feature Importance (Top 20 Principal Components)")
    importance = model.feature_importance()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[:20], y=[f"PC{i+1}" for i in range(20)])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Principal Component")
    st.pyplot()

