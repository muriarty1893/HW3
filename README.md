# Perceptron Learning Algorithm for Classification

## Project Overview
This project implements the Perceptron Learning Algorithm to solve a binary classification task. The dataset used consists of training and test data provided in an Excel file (`DataForPerceptron.xlsx`). The training data (`TRAINData` sheet) is used to train the model, and the test data (`TESTData` sheet) is used to evaluate the model's performance.

## Project Structure
- **Data Loading**: The training and testing data are read from an Excel file using `pandas`.
- **Feature and Label Separation**: The features and labels are extracted from the loaded data, and labels are converted from 0s to -1s for compatibility with the Perceptron algorithm.
- **Perceptron Class Implementation**: The perceptron algorithm is implemented from scratch with methods for training (`fit`) and predicting (`predict`).
- **Model Training**: The perceptron model is trained on the training data.
- **Model Evaluation**: The model's predictions are compared with actual test labels (if available) to compute accuracy.

## Results
- **Training Accuracy**: The accuracy of the model on the training data.
- **Test Accuracy**: The accuracy of the model on the test data if labels are present.
- **Predictions**: The predicted class labels for the test data.

## How to Run
1. Ensure the required libraries (`pandas`, `numpy`, etc.) are installed.
2. Place `DataForPerceptron.xlsx` in the working directory.
3. Run the Python script.

## Conclusion
This project demonstrates the implementation of a basic Perceptron Learning Algorithm for binary classification tasks. The results highlight the model's performance in training and testing phases and provide predictions on unseen data.
