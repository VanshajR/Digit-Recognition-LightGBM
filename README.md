
# Handwritten Digit Recognition with LightGBM

This project implements a machine learning model for handwritten digit recognition using the MNIST dataset and LightGBM. The model is deployed as a web app using Streamlit, where users can upload their own handwritten digits for recognition. The app also includes a set of sample images from the MNIST dataset in PNG format, making it easier for users to test the application.

Hosted on Streamlit: 

## LightGBM Overview

LightGBM (Light Gradient Boosting Machine) is a powerful, efficient, and scalable implementation of gradient boosting developed by Microsoft. It is widely used in machine learning tasks, especially for classification and regression problems, due to its speed and memory efficiency, making it ideal for large datasets. Below is a brief overview of its key features and advantages:

### Key Features:

- Gradient Boosting Framework: LightGBM builds an ensemble of decision trees using a boosting approach, where each tree tries to correct the errors (residuals) from the previous one.
- Leaf-wise Tree Growth: Unlike traditional level-wise tree growth methods, LightGBM uses leaf-wise growth, which results in deeper trees and often better model performance.
- Histogram-based Training: It uses histograms to discretize continuous feature values, significantly speeding up the training process and reducing memory consumption.
- Categorical Feature Handling: LightGBM has built-in support for categorical features, so they don't need to be manually encoded.
- Scalability & Parallel Training: LightGBM can efficiently scale across large datasets by training in parallel across multiple cores or even using GPUs, reducing training time considerably.
- Early Stopping: The model includes early stopping capabilities to prevent overfitting by halting training if the model performance on a validation set ceases to improve.

### Advantages of LightGBM:

- Speed & Efficiency: It is one of the fastest gradient boosting implementations, particularly suited for handling large datasets.
- Memory Efficiency: By using histogram-based methods and supporting parallel and GPU-based training, LightGBM consumes much less memory.
- High Accuracy: Its leaf-wise growth method tends to result in a more accurate model compared to traditional gradient boosting models.
- Versatility: It can be used for classification, regression, ranking, and other tasks, making it highly versatile for various machine learning applications.
- Applications of LightGBM in this Project: In this project, we use LightGBM to classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0-9) based on pixel features extracted from the images. The combination of LightGBM's fast training capabilities and its ability to handle large datasets makes it an ideal choice for this task, ensuring high accuracy in digit recognition.



## Features

- **User Upload**: Users can upload images of handwritten digits, and the model will classify them.
- **Dimensionality Reduction**: The images are transformed using PCA (Principal Component Analysis) to reduce the dimensionality, improving model performance and training speed.
- **Visualization**: After prediction, the app provides a confusion matrix and classification report to visualize the model’s performance.
- **Model Performance**: The model uses the MNIST dataset, split into training and validation sets, to predict the digit in a given image.

## Setup and Installation

### 1. Clone the repository:

```bash
git clone https://github.com/VanshajR/Digit-Recognition-LightGBM.git
cd Digit-Recognition-LightGBM
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app:

```bash
streamlit run app.py
```

This will launch the app in your browser, where you can interact with the model and upload your own handwritten digit images for classification.

## Files Overview

The following table summarizes the contents of the repository:

| File/Folder                | Description                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------|
| **`app.py`**                | The main Streamlit app file that runs the user interface and model prediction.               |
| **`train.csv` and `test.csv`** | MNIST dataset files used to train and test the model.                                          |
| **`digit_recognition_model.pkl`** | Saved LightGBM model file after training, used for prediction in the app.                      |
| **`pca_model.pkl`**         | Saved PCA model file used to transform images before feeding them into the model.             |
| **`requirements.txt`**      | Contains all the dependencies required to run the project.                                   |
| **`mnist_png/`**            | Folder containing PNG images of digits from the MNIST dataset for users to test the application. |

## How It Works

1. **Dataset Loading**: The MNIST dataset is loaded into a pandas DataFrame. The features (pixel values) are normalized to be between 0 and 1.
2. **Dimensionality Reduction**: PCA is applied to reduce the dimensionality of the dataset, making the model training faster and more efficient.
3. **Model Training**: The data is split into training and validation sets. A LightGBM model is trained on the training set to classify digits.
4. **Prediction**: Users can upload an image of a handwritten digit, and the app will preprocess the image (resize, convert to grayscale, and apply PCA) before feeding it to the trained LightGBM model for prediction.
5. **Evaluation**: The model’s accuracy and performance metrics (like classification report and confusion matrix) are displayed in the app.

### Preprocessing the Image

- The uploaded image is resized to 28x28 pixels (matching MNIST format).
- The image is converted to grayscale to reduce the complexity (as MNIST images are grayscale).
- PCA is applied to reduce the dimensionality of the image, matching the number of features used during training.

## Model Performance

Once a user uploads an image, the app provides:

- **Prediction**: Displays the predicted digit.
- **Confusion Matrix**: Shows the confusion matrix for model evaluation.
- **Classification Report**: Displays precision, recall, f1-score, and accuracy for each class (digit).
  
### Example Output

When an image is uploaded, the output will look something like this:

```
Predicted Label: 3
Classification Report:
              precision    recall  f1-score   support
          0       0.99      0.99      0.99       980
          1       0.99      0.99      0.99      1135
          2       0.99      0.99      0.99      1032
          3       0.98      0.98      0.98      1010
          4       0.99      0.99      0.99       982
          5       0.98      0.98      0.98       892
          6       0.99      0.99      0.99       958
          7       0.98      0.98      0.98      1028
          8       0.98      0.98      0.98       974
          9       0.98      0.98      0.98      1009
    accuracy                           0.99      10000
   macro avg       0.99      0.99      0.99      10000
weighted avg       0.99      0.99      0.99      10000
```



## Credits

- The MNIST dataset is publicly available from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).
- LightGBM: [LightGBM GitHub](https://github.com/microsoft/LightGBM)
