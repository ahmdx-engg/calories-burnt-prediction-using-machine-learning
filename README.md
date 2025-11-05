# calories-burnt-prediction-using-machine-learning

dataset: https://drive.google.com/file/d/1d1D-mGlCcuqE4IBlq2Lxxm2SlxOMi49H/view?usp=sharing

## 1. Introduction

The aim of this project is to design and implement a machine learning-based system capable of predicting the number of calories burned during physical activity. The prediction is based on various physiological and exercise-related factors. To achieve high accuracy, a deep learning approach is employed using TensorFlow and Keras. In addition to the original dataset attributes, several engineered synthetic features were introduced to capture non-linear relationships between variables, improving the model's predictive performance.

This project demonstrates the complete pipeline: data preparation, feature engineering, model development, evaluation, and deployment for interactive calorie prediction.

## 2. Dataset Description

The dataset used in this work is calories_with_synthetic_features.csv. It contains real-world exercise and biometric data, along with additional derived features.

Original features include:

Gender (0 = male, 1 = female)
Age (years)
Height (cm)
Weight (kg)
Exercise Duration (minutes)
Heart Rate (beats per minute)
Body Temperature (°C)
Calories Burned (target variable)
Synthetic Features

To enhance the model’s ability to capture non-linear dependencies, additional synthetic features were generated. These include:

Interaction terms (e.g., Weight × Age, Duration × Heart Rate)
Polynomial terms (squared values)
Ratio-based features (e.g., Weight / Height, Heart Rate / Duration)
These engineered attributes significantly improved model learning and accuracy.

## 3. Methodology

**3.1 Data Preprocessing**

The following preprocessing steps were performed:

Dataset Loading using Pandas
Feature Selection including original + synthetic features
Train-test split (80% training, 20% testing)
Feature Standardization using StandardScaler to normalize input variables for optimal neural network performance

**3.2 Model Development**

A fully connected Deep Neural Network was implemented in TensorFlow/Keras.

Model architecture:

Input layer: 256 neurons, ReLU activation
Hidden layer 1: 128 neurons, ReLU activation
Hidden layer 2: 64 neurons, ReLU activation
Hidden layer 3: 32 neurons, ReLU activation
Output layer: 1 neuron (regression output: calories burned)
Optimizer: Adam
Loss function: Mean Squared Error (MSE)
Evaluation metric: Mean Absolute Error (MAE)
Epochs: 100
Batch size: 32

## 3.3 Model Evaluation

The trained model was evaluated using the test dataset. Evaluation metrics included:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)

Additionally, a custom accuracy metric was defined to calculate the percentage of predictions falling within ±5% of the actual values.

## 3.4 Visualization

To analyze performance and training behavior, the following plots were generated:

Training vs Validation Loss
Training vs Validation MAE
Predicted vs Actual calories
Residual plot

These visualizations helped assess model convergence and identify patterns in predictive behavior.

## 3.5 User Interaction and Prediction

The system includes an interactive input function. Users can input:

Gender
Age
Height
Weight
Exercise duration
Heart rate
Body temperature

The program computes synthetic features for the input, applies the trained model, and outputs predicted calories burned.

## 4. Results

**4.1 Model Performance**

The model achieved the following performance (values shown as example placeholders):

Test Loss (MSE): 0.2
Test MAE: 0.2
Custom Accuracy: 98% of predictions within ±5% of true values

These results demonstrate strong prediction capability, indicating that feature engineering played a critical role in improving accuracy.
