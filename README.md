# Handwriting recognition using CNN

## Introduction

This project implements a Convolutional Neural Network (CNN) for real-time image classification using a webcam. The model is trained on a dataset of images, and the trained model is used to predict the class of objects seen by the webcam. Predictions are displayed on the screen and saved to a text file for further analysis.

## Installation

To set up the project, ensure you have Python 3.x installed. You can install the required packages using the following command:

```bash
pip install requirements.txt

**FEATURE**
***********
Real-time Image Classification: Classify images in real-time using your webcam.
Preprocessing: Convert images to grayscale, equalize the histogram, and normalize pixel values.
Training: Train a CNN model on a custom dataset of images.
Prediction Logging: Save real-time predictions to a text file for further analysis.
Visualization: Plot and visualize the average prediction probabilities for each class.

Usage
Training the Model

Train the Model: Run the train_model.py script to train the model:

This script will:

Load the images from the dataset.
Preprocess the images.
Split the data into training, validation, and test sets.
Train the CNN model.
Save the trained model as model_trained.h5.
Plot and display the training history.

Testing the Model in Real-Time
Test the Model: Run the test_model.py script to test the trained model using a webcam:
This script will:

Load the trained model from model_trained.h5.
Open the webcam and capture frames.
Preprocess each frame and use the model to predict the class.
Display the prediction on the screen.
Save the predictions to predictions.txt

## Methodology
## Data Preprocessing
    .Grayscale Conversion: Convert images to grayscale.
    .Histogram Equalization: Apply histogram equalization to enhance contrast. 
    .Normalization: Normalize pixel values to the range [0, 1].
## Model Architecture
    .Convolutional Layers: Multiple convolutional layers with ReLU activation and max-pooling.
    .Dense Layers: Fully connected layers with dropout for regularization.
    .Output Layer: Softmax activation for classification.
## Training
    .Data Augmentation: Apply random transformations to the training data to prevent overfitting.
    .Loss Function: Categorical cross-entropy.
    .Optimizer: Adam optimizer with a learning rate of 0.001.
## Real-Time Prediction
    .Capture Frame: Capture frames from the webcam.
    .Preprocess Frame: Apply the same preprocessing steps used during training.
    .Predict Class: Use the trained model to predict the class of the preprocessed frame.
    .Display Result: Show the predicted class and probability on the screen.
    .Save Predictions: Log predictions to predictions.txt.
