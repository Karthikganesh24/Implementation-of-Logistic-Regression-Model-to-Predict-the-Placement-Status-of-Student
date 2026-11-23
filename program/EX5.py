import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create dataset
data = {
    'CGPA': [6.5, 7.0, 8.5, 9.0, 6.0, 7.5, 8.0, 5.5, 9.5, 6.8],
    'IQ': [90, 100, 120, 130, 85, 110, 115, 80, 140, 95],
    'Placement': [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[['CGPA', 'IQ']]
y = df['Placement']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict placement status
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Student Placement Prediction")
print("----------------------------")
print("Actual Placement:", list(y_test))
print("Predicted Placement:", list(y_pred))
print("Model Accuracy:", accuracy * 100, "%")

# Predict for a new student
new_student = pd.DataFrame([[8.2, 125]], columns=['CGPA', 'IQ'])

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Placement Status: Placed")
else:
    print("Placement Status: Not Placed")

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
