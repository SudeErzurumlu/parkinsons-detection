# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('dataset.csv')

# Drop non-numeric columns and define the target variable (status)
X = data.drop(columns=['name', 'status'])  # Dropping the 'name' column as it contains string values
y = data['status']  # Target variable

# Check the type of the target variable, and convert if necessary
print(y.value_counts())  # Print the count of each class in the target variable
y = y.astype(int)  # Convert the target variable to integer type

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()  # Initialize the scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the testing data using the same scaler

# 1. Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)  # Initialize the Random Forest Classifier
rfc.fit(X_train_scaled, y_train)  # Train the model on the scaled training data
y_pred_rfc = rfc.predict(X_test_scaled)  # Make predictions on the scaled testing data

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rfc))  # Print accuracy of the model
print(confusion_matrix(y_test, y_pred_rfc))  # Print confusion matrix
print(classification_report(y_test, y_pred_rfc))  # Print classification report including precision, recall, and F1-score
