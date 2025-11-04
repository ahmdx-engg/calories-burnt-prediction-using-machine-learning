import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('calories_with_synthetic_features.csv')

# Define features and target variable
X = data[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
          'Weight_Age', 'Height_Weight', 'Duration_HeartRate', 'Temp_Age', 'Weight_HeartRate',
          'Height_Duration', 'Age_HeartRate', 'Body_Temp_HeartRate', 'Age_Squared', 'Weight_Squared',
          'Height_Squared', 'Duration_Squared', 'HeartRate_Squared', 'Body_Temp_Squared',
          'Weight_Age_Ratio', 'Height_Age_Ratio', 'HeartRate_Age_Ratio', 'Body_Temp_Age_Ratio',
          'Weight_Height_Ratio', 'HeartRate_Weight_Ratio', 'HeartRate_Duration_Ratio',
          'Body_Temp_Weight_Ratio', 'Body_Temp_Height_Ratio', 'Duration_Weight_Ratio',
          'Duration_Height_Ratio', 'HeartRate_Duration', 'Body_Temp_Duration', 'Weight_Duration',
          'Height_Duration', 'Age_Weight', 'Age_Height']].values
y = data['Calories'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1)  # Output layer
])
learning_rate = 0.0001

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
              loss='mean_squared_error', 
              metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Make predictions
predictions = model.predict(X_test)

# Custom accuracy function
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    return np.mean(np.abs(y_true - y_pred) / y_true < tolerance) * 100

# Calculate accuracy
accuracy = calculate_accuracy(y_test, predictions.flatten())
print(f'Custom Accuracy: {accuracy:.2f}%')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.title('Predicted vs Actual Calories')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.axis('equal')
plt.grid()
plt.show()

residuals = y_test - predictions.flatten()

plt.figure(figsize=(12, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted Calories')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# Function to calculate synthetic features
# Function to calculate synthetic features
def calculate_synthetic_features(inputs):
    gender, age, height, weight, duration, heart_rate, body_temp = inputs
    synthetic_features = [
        weight * age,  # Weight_Age
        height * weight,  # Height_Weight
        duration * heart_rate,  # Duration_HeartRate
        body_temp * age,  # Temp_Age
        weight * heart_rate,  # Weight_HeartRate
        height * duration,  # Height_Duration
        age * heart_rate,  # Age_HeartRate
        body_temp * heart_rate,  # Body_Temp_HeartRate
        age ** 2,  # Age_Squared
        weight ** 2,  # Weight_Squared
        height ** 2,  # Height_Squared
        duration ** 2,  # Duration_Squared
        heart_rate ** 2,  # HeartRate_Squared
        body_temp ** 2, # Body_Temp_Squared
        weight / age if age != 0 else 0,  # Weight_Age_Ratio
        height / age if age != 0 else 0,  # Height_Age_Ratio
        heart_rate / age if age != 0 else 0,  # HeartRate_Age_Ratio
        body_temp / age if age != 0 else 0,  # Body_Temp_Age_Ratio
        weight / height if height != 0 else 0,  # Weight_Height_Ratio
        heart_rate / weight if weight != 0 else 0,  # HeartRate_Weight_Ratio
        heart_rate / duration if duration != 0 else 0,  # HeartRate_Duration_Ratio
        body_temp / weight if weight != 0 else 0,  # Body_Temp_Weight_Ratio
        body_temp / height if height != 0 else 0,  # Body_Temp_Height_Ratio
        duration / weight if weight != 0 else 0,  # Duration_Weight_Ratio
        duration / height if height != 0 else 0,  # Duration_Height_Ratio
        heart_rate * duration,  # HeartRate_Duration
        body_temp * duration,  # Body_Temp_Duration
        weight * duration,  # Weight_Duration
        height * duration,  # Height_Duration
        age * weight,  # Age_Weight
        age * height  # Age_Height
    ]
    return synthetic_features
# Function to get user input
def get_user_input():
    gender = int(input("Enter Gender (0 for Male, 1 for Female): "))
    age = float(input("Enter Age: "))
    height = float(input("Enter Height (in cm): "))
    weight = float(input("Enter Weight (in kg): "))
    duration = float(input("Enter Duration of exercise (in minutes): "))
    heart_rate = float(input("Enter Heart Rate (in bpm): "))
    body_temp = float(input("Enter Body Temperature (in °C): "))

    user_input = np.array([gender, age, height, weight, duration, heart_rate, body_temp])
    synthetic_features = calculate_synthetic_features(user_input)
    full_input = np.concatenate((user_input, synthetic_features))
    user_input_scaled = scaler.transform(full_input.reshape(1, -1))
    
    return user_input_scaled

def predict_calories(user_input):
    prediction = model.predict(user_input)
    return prediction[0][0]

def main():
    user_input = get_user_input()
    predicted_calories = predict_calories(user_input)
    print(f"Predicted Calories Burned: {predicted_calories:.2f}")

if __name__ == "__main__":
    main()
