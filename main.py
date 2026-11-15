
# Student Score Prediction
# Machine Learning with Python

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

scores = np.array([20, 30, 40, 50, 60, 70, 80, 90])

# Create and train the model
model = LinearRegression()
model.fit(hours, scores)

# Get user input
study_hours = float(input("Enter study hours: "))

# Predict student score
prediction = model.predict([[study_hours]])

print("Predicted score:", round(prediction[0], 2))
