# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
X_train = pd.read_csv("data/preprocessed/X_train.csv")
X_test = pd.read_csv("data/preprocessed/X_test.csv")
y_train = pd.read_csv("data/preprocessed/y_train.csv")
y_test = pd.read_csv("data/preprocessed/y_test.csv")

# Convert target to 1D array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(model, "models/logistic_model.pkl")

print("Model training completed")
print("Accuracy:", accuracy)