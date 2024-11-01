# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('dataset.csv')

# Filter out rows with target values other than 0 or 1
data = data[data['status'].isin([0, 1])]

# Prepare the data
X = data.drop(columns=['name', 'status'])  # Drop non-numeric column 'name'
y = data['status']  # Target variable

# Print the value counts of the target variable
print(y.value_counts())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rfc = rfc.predict(X_test_scaled)

# Print the evaluation metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rfc))
print(confusion_matrix(y_test, y_pred_rfc))
print(classification_report(y_test, y_pred_rfc))
